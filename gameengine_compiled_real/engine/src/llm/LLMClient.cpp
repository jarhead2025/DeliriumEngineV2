// FILEPATH: engine/src/llm/LLMClient.cpp
// POSIX-socket HTTP client for a local Ollama-compatible LLM server.
#include "delirium/llm/LLMClient.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <sstream>

// WHY NO nlohmann: CMake declares nlohmann_json OPTIONAL, but this file
// sits in the engine/src/*.cpp GLOB -- so an unconditional include would
// break the WHOLE engine lib (and every CTest gate) on a machine without
// it, not just the LLM client. The engine already ships a strict,
// dependency-free JSON reader (ai/ElementTreeParser.h); this file uses it.
#include "delirium/ai/ElementTreeParser.h"

namespace {
// Escape a string for embedding in a JSON document.
std::string JsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (const char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// The assistant text at choices[0].message.content (or .delta.content).
// Returns "" when the body is not a chat-completions response.
std::string ChatTextFrom(const std::string& body) {
    DeliRium::AI::JsonValue j;
    std::string why;
    if (!DeliRium::AI::ParseJson(body, j, why)) return {};
    const DeliRium::AI::JsonValue* choices = j.Find("choices");
    if (!choices || !choices->IsArray() || choices->items.empty()) return {};
    const DeliRium::AI::JsonValue& c = choices->items[0];
    const DeliRium::AI::JsonValue* msg = c.Find("message");
    if (msg) {
        const DeliRium::AI::JsonValue* content = msg->Find("content");
        if (content && content->IsString()) return content->text;
    }
    const DeliRium::AI::JsonValue* delta = c.Find("delta");
    if (delta) {
        const DeliRium::AI::JsonValue* content = delta->Find("content");
        if (content && content->IsString()) return content->text;
    }
    return {};
}
} // namespace

namespace DeliRium {
namespace LLM {

namespace {

// Connect to host:port. Returns the socket fd or -1.
int ConnectTo(const std::string& host, int port, int timeout_seconds,
               std::string* error) {
    struct addrinfo hints {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* res = nullptr;
    const std::string port_s = std::to_string(port);
    const int rc = getaddrinfo(host.c_str(), port_s.c_str(), &hints, &res);
    if (rc != 0 || !res) {
        if (error) *error = std::string("getaddrinfo failed: ") + gai_strerror(rc);
        return -1;
    }
    int fd = -1;
    for (struct addrinfo* ai = res; ai; ai = ai->ai_next) {
        fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        struct timeval tv {};
        tv.tv_sec = timeout_seconds;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        if (::connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        ::close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0 && error) *error = "connect failed";
    return fd;
}

bool WriteAll(int fd, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        const ssize_t n = ::send(fd, data + sent, len - sent, MSG_NOSIGNAL);
        if (n <= 0) {
            if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
            return false;
        }
        sent += size_t(n);
    }
    return true;
}

// Read the whole response body. Handles chunked transfer-encoding (which
// Ollama's SSE endpoint uses) and Content-Length.
std::string ReadResponse(int fd, bool* ok, std::string* error) {
    std::string raw;
    char buf[8192];
    // --- headers ---
    size_t header_end = std::string::npos;
    while (true) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            if (error) *error = "connection closed while reading headers";
            if (ok) *ok = false;
            return raw;
        }
        raw.append(buf, size_t(n));
        header_end = raw.find("\r\n\r\n");
        if (header_end != std::string::npos) break;
        if (raw.size() > 1 << 20) {
            if (error) *error = "header too large";
            if (ok) *ok = false;
            return raw;
        }
    }
    const std::string headers = raw.substr(0, header_end);
    std::string body = raw.substr(header_end + 4);

    const bool chunked = headers.find("Transfer-Encoding: chunked") != std::string::npos ||
                         headers.find("transfer-encoding: chunked") != std::string::npos;

    if (!chunked) {
        // Content-Length body (or "read until close").
        size_t clen = 0;
        bool have_clen = false;
        const size_t cl = headers.find("Content-Length:");
        if (cl != std::string::npos) {
            clen = size_t(std::strtoul(headers.c_str() + cl + 15, nullptr, 10));
            have_clen = true;
        }
        while (true) {
            if (have_clen && body.size() >= clen) break;
            const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
            if (n <= 0) {
                if (n < 0 && errno == EINTR) continue;
                break;
            }
            body.append(buf, size_t(n));
        }
        if (have_clen && body.size() > clen) body.resize(clen);
        if (ok) *ok = true;
        return body;
    }

    // --- chunked body ---
    // Format: <hex-size>\r\n<data>\r\n ... 0\r\n\r\n
    // Decoded chunks are appended; the SSE framing ("data: ...\n\n") lives
    // INSIDE the chunk data, so we keep the decoded bytes as-is.
    std::string decoded;
    size_t pos = 0;
    while (true) {
        // Need a full size line.
        size_t eol = body.find("\r\n", pos);
        while (eol == std::string::npos) {
            const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
            if (n <= 0) {
                if (n < 0 && errno == EINTR) continue;
                if (error) *error = "connection closed mid-chunk";
                if (ok) *ok = false;
                return decoded;
            }
            body.append(buf, size_t(n));
            eol = body.find("\r\n", pos);
        }
        const unsigned long csize = std::strtoul(body.c_str() + pos, nullptr, 16);
        pos = eol + 2;
        if (csize == 0) { if (ok) *ok = true; return decoded; }
        while (body.size() < pos + csize + 2) {
            const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
            if (n <= 0) {
                if (n < 0 && errno == EINTR) continue;
                if (error) *error = "connection closed mid-chunk-data";
                if (ok) *ok = false;
                return decoded;
            }
            body.append(buf, size_t(n));
        }
        decoded.append(body, pos, csize);
        pos += csize + 2;   // skip the trailing CRLF
    }
}

// A simple HTTP GET. Returns the body.
std::string HttpGet(const std::string& host, int port, const std::string& path,
                    int timeout_seconds, bool* ok, std::string* error) {
    const int fd = ConnectTo(host, port, timeout_seconds, error);
    if (fd < 0) { if (ok) *ok = false; return {}; }
    std::string req = "GET " + path + " HTTP/1.1\r\n";
    req += "Host: " + host + ":" + std::to_string(port) + "\r\n";
    req += "User-Agent: Delirium/1.0\r\n";
    req += "Accept: application/json\r\n";
    req += "Connection: close\r\n\r\n";
    if (!WriteAll(fd, req.data(), req.size())) {
        if (error) *error = "send failed";
        if (ok) *ok = false;
        ::close(fd);
        return {};
    }
    std::string body = ReadResponse(fd, ok, error);
    ::close(fd);
    return body;
}

} // namespace

// --- payload building ------------------------------------------------------
std::string BuildChatPayload(const Config& cfg,
                              const std::string& user_message,
                              const std::vector<std::pair<std::string, std::string>>& history) {
    // Hand-built JSON (the in-tree reader is a SUBSET reader -- it parses
    // but does not serialize -- so the request body is assembled directly).
    std::string body = "{\"model\":\"";
    body += JsonEscape(cfg.model);
    body += "\",\"messages\":[";
    bool first = true;
    auto push_msg = [&](const char* role, const std::string& content) {
        if (!first) body += ",";
        first = false;
        body += "{\"role\":\"";
        body += role;
        body += "\",\"content\":\"";
        body += JsonEscape(content);
        body += "\"}";
    };
    if (!cfg.system.empty()) push_msg("system", cfg.system);
    for (const auto& kv : history) push_msg(kv.first.c_str(), kv.second);
    push_msg("user", user_message);
    body += "],\"stream\":true,\"temperature\":";
    body += std::to_string(cfg.temperature);
    body += ",\"max_tokens\":";
    body += std::to_string(cfg.max_tokens);
    body += "}";
    return body;
}

std::string ExtractChatText(const std::string& body) {
    return ChatTextFrom(body);
}

std::string ExtractDeltaText(const std::string& chunk_json) {
    if (chunk_json.rfind("[DONE]", 0) == 0) return {};
    return ChatTextFrom(chunk_json);
}

// --- client ----------------------------------------------------------------
LLMClient::LLMClient(const Config& cfg) : cfg_(cfg) {}

LLMClient::~LLMClient() {
    Cancel();
    if (worker_.joinable()) worker_.join();
}

bool LLMClient::Send(const std::string& user_message,
                     const std::vector<std::pair<std::string, std::string>>& history) {
    if (IsBusy()) return false;
    if (worker_.joinable()) worker_.join();   // reap the previous one
    {
        std::lock_guard<std::mutex> lk(mu_);
        partial_.clear();
        final_.clear();
        error_.clear();
        has_response_ = false;
    }
    cancel_ = false;
    state_.store(State::Connecting, std::memory_order_release);
    const std::string payload = BuildChatPayload(cfg_, user_message, history);
    worker_ = std::thread([this, payload]() { Run(payload); });
    return true;
}

void LLMClient::Cancel() {
    cancel_ = true;
}

std::string LLMClient::Partial() const {
    std::lock_guard<std::mutex> lk(mu_);
    return partial_;
}

std::string LLMClient::TakeResponse() {
    std::lock_guard<std::mutex> lk(mu_);
    std::string out = final_;
    final_.clear();
    has_response_ = false;
    if (state_.load(std::memory_order_acquire) == State::Done)
        state_.store(State::Idle, std::memory_order_release);
    return out;
}

std::string LLMClient::Error() const {
    std::lock_guard<std::mutex> lk(mu_);
    return error_;
}

void LLMClient::Run(const std::string& payload) {
    std::string err;
    const int fd = ConnectTo(cfg_.host, cfg_.port, cfg_.timeout_seconds, &err);
    if (fd < 0) {
        std::lock_guard<std::mutex> lk(mu_);
        error_ = "cannot reach " + cfg_.host + ":" + std::to_string(cfg_.port) +
                 " (" + err + "). Is the local LLM server running?";
        state_.store(State::Failed, std::memory_order_release);
        return;
    }
    state_.store(State::Streaming, std::memory_order_release);

    std::string req = "POST /v1/chat/completions HTTP/1.1\r\n";
    req += "Host: " + cfg_.host + ":" + std::to_string(cfg_.port) + "\r\n";
    req += "User-Agent: Delirium/1.0\r\n";
    req += "Content-Type: application/json\r\n";
    req += "Accept: text/event-stream\r\n";
    req += "Content-Length: " + std::to_string(payload.size()) + "\r\n";
    req += "Connection: close\r\n\r\n";
    req += payload;
    if (!WriteAll(fd, req.data(), req.size())) {
        ::close(fd);
        std::lock_guard<std::mutex> lk(mu_);
        error_ = "failed to send the request";
        state_.store(State::Failed, std::memory_order_release);
        return;
    }

    // Read the response incrementally so the UI can paint tokens as they
    // arrive. ReadResponse() would block until the whole answer is done --
    // we want the streaming behaviour.
    std::string raw;
    char buf[8192];
    size_t header_end = std::string::npos;
    bool chunked = false;
    std::string body;
    std::string decoded;

    // --- headers ---
    while (true) {
        if (cancel_.load()) { ::close(fd); state_.store(State::Idle, std::memory_order_release); return; }
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            break;
        }
        raw.append(buf, size_t(n));
        header_end = raw.find("\r\n\r\n");
        if (header_end != std::string::npos) break;
    }
    if (header_end == std::string::npos) {
        ::close(fd);
        std::lock_guard<std::mutex> lk(mu_);
        error_ = "no response headers";
        state_.store(State::Failed, std::memory_order_release);
        return;
    }
    const std::string headers = raw.substr(0, header_end);
    body = raw.substr(header_end + 4);
    chunked = headers.find("chunked") != std::string::npos;

    // A pending SSE buffer: lines are "data: {json}" separated by "\n\n".
    std::string pending;

    auto consume_sse = [&](const std::string& text) {
        pending += text;
        size_t p = 0;
        while (true) {
            // SSE events are separated by a blank line; but Ollama emits one
            // "data:" line per event followed by "\n\n". Handle both.
            size_t eol = pending.find('\n', p);
            if (eol == std::string::npos) break;
            std::string line = pending.substr(p, eol - p);
            p = eol + 1;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.rfind("data:", 0) != 0) continue;
            std::string payload_line = line.substr(5);
            while (!payload_line.empty() && payload_line.front() == ' ') payload_line.erase(0, 1);
            if (payload_line == "[DONE]") {
                std::lock_guard<std::mutex> lk(mu_);
                final_ = partial_;
                has_response_ = true;
                state_.store(State::Done, std::memory_order_release);
                return true;
            }
            const std::string piece = ExtractDeltaText(payload_line);
            if (!piece.empty()) {
                std::lock_guard<std::mutex> lk(mu_);
                partial_ += piece;
            }
        }
        pending.erase(0, p);
        return false;
    };

    if (!chunked) {
        // Non-streaming body (the server ignored "stream": true).
        while (true) {
            if (cancel_.load()) { ::close(fd); state_.store(State::Idle, std::memory_order_release); return; }
            const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
            if (n <= 0) { if (n < 0 && errno == EINTR) continue; break; }
            body.append(buf, size_t(n));
        }
        const std::string text = ExtractChatText(body);
        {
            std::lock_guard<std::mutex> lk(mu_);
            partial_ = text;
            final_ = text;
            has_response_ = true;
        }
        ::close(fd);
        state_.store(State::Done, std::memory_order_release);
        return;
    }

    // --- chunked stream ---
    size_t pos = 0;
    while (true) {
        if (cancel_.load()) { ::close(fd); state_.store(State::Idle, std::memory_order_release); return; }
        size_t eol = body.find("\r\n", pos);
        while (eol == std::string::npos) {
            const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
            if (n <= 0) {
                if (n < 0 && errno == EINTR) continue;
                ::close(fd);
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    if (partial_.empty()) error_ = "the model returned nothing";
                    else { final_ = partial_; has_response_ = true; }
                }
                state_.store(state_.load(std::memory_order_acquire) == State::Failed
                                 ? State::Failed : State::Done,
                             std::memory_order_release);
                return;
            }
            body.append(buf, size_t(n));
            eol = body.find("\r\n", pos);
        }
        const unsigned long csize = std::strtoul(body.c_str() + pos, nullptr, 16);
        pos = eol + 2;
        if (csize == 0) break;   // end of the chunked body
        while (body.size() < pos + csize + 2) {
            const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
            if (n <= 0) {
                if (n < 0 && errno == EINTR) continue;
                break;
            }
            body.append(buf, size_t(n));
        }
        const std::string chunk = body.substr(pos, csize);
        pos += csize + 2;
        if (consume_sse(chunk)) { ::close(fd); return; }
    }
    ::close(fd);
    {
        std::lock_guard<std::mutex> lk(mu_);
        final_ = partial_;
        has_response_ = true;
        if (final_.empty()) error_ = "the model returned nothing";
    }
    state_.store(final_.empty() ? State::Failed : State::Done, std::memory_order_release);
}

// --- discovery -------------------------------------------------------------
namespace {
// Collect the string member `key` of every object in the array `arr_key`.
bool CollectIds(const std::string& body, const char* arr_key, const char* id_key,
                std::vector<std::string>* out, std::string* error) {
    DeliRium::AI::JsonValue j;
    std::string why;
    if (!DeliRium::AI::ParseJson(body, j, why)) {
        if (error) *error = "bad JSON: " + why;
        return false;
    }
    const DeliRium::AI::JsonValue* arr = j.Find(arr_key);
    if (!arr || !arr->IsArray()) {
        if (error) *error = std::string("unexpected response (no ") + arr_key + " array)";
        return false;
    }
    for (const DeliRium::AI::JsonValue& m : arr->items) {
        const DeliRium::AI::JsonValue* id = m.Find(id_key);
        if (id && id->IsString()) out->push_back(id->text);
    }
    return true;
}
} // namespace

bool LLMClient::ListModels(const std::string& host, int port,
                           std::vector<std::string>* out, std::string* error) {
    bool ok = false;
    const std::string body = HttpGet(host, port, "/v1/models", 10, &ok, error);
    if (!ok) return false;
    return CollectIds(body, "data", "id", out, error);
}

bool LLMClient::ListOllamaModels(const std::string& host, int port,
                                 std::vector<std::string>* out, std::string* error) {
    bool ok = false;
    const std::string body = HttpGet(host, port, "/api/tags", 10, &ok, error);
    if (!ok) return false;
    return CollectIds(body, "models", "name", out, error);
}

bool LLMClient::Ping(const std::string& host, int port, int timeout_seconds) {
    std::string err;
    bool ok = false;
    HttpGet(host, port, "/v1/models", timeout_seconds, &ok, &err);
    return ok;
}

} // namespace LLM
} // namespace DeliRium
