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
#include <chrono>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <thread>
#include <utility>   // std::pair (the payload builders' history type)

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

// The Content-Length VALUE from a header block, or 0 if there is none.
//
// WHY A HELPER: the offset must skip the whole header name AND its colon
// AND any spaces. "Content-Length:" is exactly 15 characters, so
// `find(...) + 15` points AT the colon -- and `strtoul(": 412")` returns 0.
//
// MEASURED: that off-by-one made the client's native-endpoint path read
// only the first packet of the body, fail to parse the truncated JSON, and
// report "the model returned nothing" -- a TRANSPORT failure for a server
// that had answered perfectly, after the model had spent MINUTES
// generating. The GET path survived it only because it falls back to
// "read until close" when the length is 0.
//
// One definition, so the two call sites cannot drift again.
size_t ContentLengthOf(const std::string& headers) {
    const size_t at = headers.find("Content-Length:");
    if (at == std::string::npos) return 0;
    // "Content-Length:" is 15 characters, so `at + 15` points AT the
    // colon; `at + 16` is the first character AFTER it.
    const char* p = headers.c_str() + at + 16;
    while (*p == ' ' || *p == '\t') ++p;
    return size_t(std::strtoul(p, nullptr, 10));
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
            clen = ContentLengthOf(headers);
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

// Decode a CHUNKED body that has already been read into memory.
//
// Format: <hex-size>\r\n<data>\r\n ... 0\r\n\r\n
//
// WHY A SEPARATE BUFFER-BASED HELPER: the socket-based decoder below needs
// to READ more bytes when a chunk is incomplete, but the native path has
// already read the whole reply. And the native endpoint's framing DEPENDS
// ON THE REPLY SIZE -- MEASURED: a short answer (413 bytes) comes back
// with "Content-Length", while a long answer (2,339 bytes) comes back
// with "Transfer-Encoding: chunked". A client that handles only one of
// them fails on exactly half the replies, and it fails as "the model
// returned nothing" -- because the chunk framing is still in the bytes
// and the JSON parse fails.
std::string DecodeChunkedBuffer(const std::string& body) {
    std::string decoded;
    size_t pos = 0;
    while (true) {
        const size_t eol = body.find("\r\n", pos);
        if (eol == std::string::npos) break;     // incomplete -> stop
        const unsigned long csize = std::strtoul(body.c_str() + pos, nullptr, 16);
        pos = eol + 2;
        if (csize == 0) return decoded;          // end of the body
        if (body.size() < pos + csize) break;     // incomplete -> stop
        decoded.append(body, pos, csize);
        pos += csize + 2;                        // skip the trailing CRLF
    }
    return decoded;
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

// The OLLAMA-NATIVE payload for /api/generate.
//
// It is NOT the OpenAI shape: a flat "prompt" (plus an optional "system"),
// and "options" for the sampling parameters.
//
// WHY IT EXISTS: the native endpoint HONOURS "keep_alive", so the model
// unloads when it answers. The OpenAI-compatible endpoint ignores it --
// MEASURED: after a request with "keep_alive": 0 the model was still
// resident. That is what makes the escalation ladder possible on a machine
// that cannot hold both models at once (see Config::use_native_endpoint).
std::string BuildNativePayload(const Config& cfg,
                               const std::string& user_message,
                               const std::vector<std::pair<std::string, std::string>>& history) {
    std::string body = "{\"model\":\"";
    body += JsonEscape(cfg.model);
    body += "\"";
    if (!cfg.system.empty()) {
        body += ",\"system\":\"";
        body += JsonEscape(cfg.system);
        body += "\"";
    }
    // The native endpoint has no multi-turn "messages" array; the history
    // is folded into the prompt as labelled turns. (The learner never sends
    // history -- the escalation prompt is SELF-CONTAINED -- so this path is
    // for completeness.)
    if (!history.empty()) {
        body += ",\"prompt\":\"";
        for (const auto& kv : history) {
            body += JsonEscape(kv.first);
            body += ": ";
            body += JsonEscape(kv.second);
            body += "\\n\\n";
        }
        body += JsonEscape(user_message);
        body += "\"";
    } else {
        body += ",\"prompt\":\"";
        body += JsonEscape(user_message);
        body += "\"";
    }
    body += ",\"stream\":false,\"keep_alive\":";
    body += std::to_string(cfg.keep_alive);
    body += ",\"options\":{\"temperature\":";
    body += std::to_string(cfg.temperature);
    body += ",\"num_predict\":";
    body += std::to_string(cfg.max_tokens);
    body += "}}";
    return body;
}

// The "response" field of a native /api/generate reply.
std::string ExtractNativeText(const std::string& body) {
    DeliRium::AI::JsonValue j;
    std::string why;
    if (!DeliRium::AI::ParseJson(body, j, why)) return {};
    const DeliRium::AI::JsonValue* r = j.Find("response");
    if (r && r->IsString()) return r->text;
    return {};
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
    // The payload shape depends on the endpoint: the OpenAI-compatible one
    // takes "messages", the native one takes a flat "prompt" (and honours
    // keep_alive).
    const std::string payload = cfg_.use_native_endpoint
        ? BuildNativePayload(cfg_, user_message, history)
        : BuildChatPayload(cfg_, user_message, history);
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

    // -----------------------------------------------------------------------
    // THE OLLAMA-NATIVE PATH (/api/generate).
    //
    // It is a SEPARATE branch because the wire protocol differs: the native
    // endpoint answers with ONE JSON body (no SSE, no chunking), and it
    // honours "keep_alive" so the model can UNLOAD when it answers -- which
    // is what lets the escalation ladder run two models SEQUENTIALLY on a
    // machine that cannot hold both at once.
    //
    // (MEASURED: /v1/chat/completions ignores keep_alive -- the model was
    // still resident after a "keep_alive": 0 request. /api/generate honours
    // it -- the model unloaded immediately.)
    // -----------------------------------------------------------------------
    if (cfg_.use_native_endpoint) {
        std::string req = "POST /api/generate HTTP/1.1\r\n";
        req += "Host: " + cfg_.host + ":" + std::to_string(cfg_.port) + "\r\n";
        req += "User-Agent: Delirium/1.0\r\n";
        req += "Content-Type: application/json\r\n";
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

        // Read the WHOLE body. The native endpoint answers in one go, so
        // there is nothing to stream -- but the read must still tolerate a
        // long silence (the server is THINKING), with the same
        // retry-on-timeout rule as the SSE path.
        //
        // THE OVERALL DEADLINE.
        //
        // `SO_RCVTIMEO` is a PER-READ timeout: it fires when a single
        // recv goes quiet, not when the whole request has gone on too
        // long. So a server that keeps sending a byte every few seconds
        // can stream FOREVER -- MEASURED: the 7B was still generating
        // after 41 minutes on a dense document, with the client dutifully
        // waiting, because every single recv succeeded.
        //
        // So the path needs a deadline on the REQUEST, not on the read.
        // `timeout_seconds` is that budget, counted from the first byte.
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::seconds(cfg_.timeout_seconds);
        std::string raw;
        char nbuf[8192];
        size_t hdr_end = std::string::npos;
        {
            const int max_waits = cfg_.timeout_seconds * 2;   // 500 ms each
            for (int waits = 0; waits < max_waits && hdr_end == std::string::npos; ++waits) {
                if (std::chrono::steady_clock::now() > deadline) {
                    ::close(fd);
                    std::lock_guard<std::mutex> lk(mu_);
                    error_ = "the request exceeded its time budget";
                    state_.store(State::Failed, std::memory_order_release);
                    return;
                }
                const ssize_t n = ::recv(fd, nbuf, sizeof(nbuf), 0);
                if (n > 0) {
                    raw.append(nbuf, size_t(n));
                    hdr_end = raw.find("\r\n\r\n");
                    if (hdr_end != std::string::npos) break;
                    continue;
                }
                if (n == 0) break;                 // the peer closed
                if (errno == EINTR) continue;
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }
        if (hdr_end == std::string::npos) {
            ::close(fd);
            std::lock_guard<std::mutex> lk(mu_);
            error_ = "no response headers";
            state_.store(State::Failed, std::memory_order_release);
            return;
        }
        // The body. The framing DEPENDS ON THE REPLY SIZE (MEASURED):
        //   - a short answer (413 bytes) -> "Content-Length: 413"
        //   - a long answer (2,339 bytes) -> "Transfer-Encoding: chunked"
        // So BOTH must be handled, and a reply with neither must be read
        // until the connection closes.
        const std::string headers = raw.substr(0, hdr_end);
        std::string body = raw.substr(hdr_end + 4);
        const bool chunked = headers.find("Transfer-Encoding: chunked") != std::string::npos ||
                             headers.find("transfer-encoding: chunked") != std::string::npos;
        const size_t cl = ContentLengthOf(headers);
        // Whether the body was CUT SHORT. Without this the over-long
        // request falls through to "the model returned nothing" -- a
        // GENERIC message that points at the model instead of at the clock.
        // MEASURED: a server that streams forever produced exactly that
        // message, even though the deadline had correctly stopped it.
        auto deadline_passed = [&] {
            return std::chrono::steady_clock::now() > deadline;
        };
        bool body_truncated = false;
        if (chunked) {
            // Read the REST of the body first (the chunks may not have all
            // arrived with the headers), then decode. The deadline applies
            // here too -- the body is where a long generation actually
            // spends its time.
            while (!deadline_passed()) {
                const ssize_t n = ::recv(fd, nbuf, sizeof(nbuf), 0);
                if (n <= 0) break;
                body.append(nbuf, size_t(n));
            }
            body_truncated = deadline_passed();
            body = DecodeChunkedBuffer(body);
        } else if (cl > body.size()) {
            // Content-Length body: read exactly that many bytes.
            std::string rest;
            rest.reserve(cl - body.size());
            while (body.size() + rest.size() < cl && !deadline_passed()) {
                const ssize_t n = ::recv(fd, nbuf, sizeof(nbuf), 0);
                if (n <= 0) break;
                rest.append(nbuf, size_t(n));
            }
            body += rest;
            body_truncated = (body.size() < cl);
        } else if (cl == 0) {
            // NO length and NOT chunked: read until the connection closes
            // (or the deadline passes). (Ollama sends "Connection: close".)
            while (!deadline_passed()) {
                const ssize_t n = ::recv(fd, nbuf, sizeof(nbuf), 0);
                if (n <= 0) break;
                body.append(nbuf, size_t(n));
            }
            body_truncated = deadline_passed();
        }
        ::close(fd);
        const std::string text = ExtractNativeText(body);
        {
            std::lock_guard<std::mutex> lk(mu_);
            partial_ = text;
            final_ = text;
            has_response_ = true;
            if (text.empty()) {
                // Name the ACTUAL cause. "the model returned nothing" points
                // at the model; a deadline points at the CLOCK.
                error_ = body_truncated
                    ? "the request exceeded its time budget"
                    : "the model returned nothing";
            }
        }
        state_.store(text.empty() ? State::Failed : State::Done,
                     std::memory_order_release);
        return;
    }

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
    //
    // The socket has an SO_RCVTIMEO, so a recv can time out (EAGAIN) while
    // the server is THINKING. A large prompt takes minutes before the
    // first byte comes back, so a timeout here must be RETRIED, not
    // treated as "no response".
    //
    // The old code broke out of the loop on any recv failure other than
    // EINTR, and then reported "no response headers" -- a TRANSPORT
    // failure for a server that was working perfectly.
    bool headers_ok = false;
    // THE OVERALL DEADLINE for the SSE path.
    //
    // `SO_RCVTIMEO` is a PER-READ timeout, so a server that keeps sending
    // a byte every few seconds can stream FOREVER. MEASURED: the 7B was
    // still generating after 41 minutes on a dense document, with the
    // client dutifully waiting, because every single recv succeeded.
    //
    // The SSE path is the WORST case for this (it is explicitly a
    // streaming protocol), so it needs the deadline most.
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(cfg_.timeout_seconds);
    {
        const int max_waits = cfg_.timeout_seconds * 2;   // 500 ms each
        for (int waits = 0; waits < max_waits && header_end == std::string::npos; ++waits) {
            if (std::chrono::steady_clock::now() > deadline) {
                ::close(fd);
                std::lock_guard<std::mutex> lk(mu_);
                error_ = "the request exceeded its time budget";
                state_.store(State::Failed, std::memory_order_release);
                return;
            }
            const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
            if (n > 0) {
                raw.append(buf, size_t(n));
                header_end = raw.find("\r\n\r\n");
                if (header_end != std::string::npos) { headers_ok = true; break; }
                continue;
            }
            if (n == 0) break;                 // the peer closed
            if (errno == EINTR) continue;
            // A read timeout: the server is still thinking. Wait and retry.
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
    if (header_end == std::string::npos) {
        ::close(fd);
        std::lock_guard<std::mutex> lk(mu_);
        error_ = headers_ok ? "malformed response" : "no response headers";
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
        // Read the CHUNK-SIZE line. It has the same sharp edge as the
        // header read: the socket has an SO_RCVTIMEO, so a recv can time
        // out (the server is still thinking) and the peer can close.
        //
        // The old code broke out on any recv failure and then set
        // error_ = "the model returned nothing" -- a GENERIC message --
        // and, because the state was not already Failed, stored
        // State::Done. So a server that closed mid-response was reported as
        // a SUCCESSFUL response with an empty body.
        //
        // Now: a timeout is retried, a close is named, and an empty
        // response is a FAILURE (an empty reply is not an answer).
        size_t eol = body.find("\r\n", pos);
        bool peer_closed = false;
        bool timed_out = false;
        {
            const int max_waits = cfg_.timeout_seconds * 2;   // 500 ms each
            for (int waits = 0; waits < max_waits && eol == std::string::npos; ++waits) {
                if (std::chrono::steady_clock::now() > deadline) {
                    timed_out = true;
                    break;
                }
                const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
                if (n > 0) {
                    body.append(buf, size_t(n));
                    eol = body.find("\r\n", pos);
                    if (eol != std::string::npos) break;
                    continue;
                }
                if (n == 0) { peer_closed = true; break; }   // the peer closed
                if (errno == EINTR) continue;
                // A read timeout: the server is still thinking. Wait.
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }
        if (eol == std::string::npos) {
            ::close(fd);
            bool have_text = false;
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (peer_closed) {
                    error_ = "the server closed the connection mid-response";
                } else if (!partial_.empty()) {
                    // Some text DID arrive before the trouble: keep it, and
                    // report the response as finished rather than failed --
                    // a partial answer is more useful than an error.
                    final_ = partial_;
                    has_response_ = true;
                } else if (timed_out) {
                    error_ = "the request exceeded its time budget";
                } else {
                    error_ = "the response timed out before the first chunk";
                }
                have_text = !partial_.empty();
            }
            state_.store(have_text ? State::Done : State::Failed,
                         std::memory_order_release);
            return;
        }
        const unsigned long csize = std::strtoul(body.c_str() + pos, nullptr, 16);
        pos = eol + 2;
        if (csize == 0) break;   // end of the chunked body
        // Read the chunk BODY. The socket has an SO_RCVTIMEO, so a recv can
        // fail with EAGAIN/EWOULDBLOCK (a TIMEOUT) rather than returning
        // data.
        //
        // WHY IT MATTERS: the old code treated any recv failure other than
        // EINTR as "the chunk is complete" and fell through to
        // `body.substr(pos, csize)`. A timeout mid-chunk therefore produced
        // a TRUNCATED chunk with garbage after it, the SSE parse found
        // nothing usable, and the request was reported as "the model
        // returned nothing" -- a TRANSPORT failure with a perfectly healthy
        // server and a perfectly good prompt.
        //
        // A timeout is not a disconnect: the server may simply be thinking
        // (a 2,900-token prompt on a loaded machine takes minutes BEFORE
        // the first token arrives). So it must RETRY, up to the socket's
        // own deadline.
        bool chunk_complete = false;
        bool recv_failed = false;
        {
            // The deadline is the WHOLE REQUEST's, not this chunk's: a
            // long generation streams many chunks, and each chunk's wait
            // must not restart the budget.
            const int max_waits = cfg_.timeout_seconds * 2;   // 500 ms each
            for (int waits = 0; waits < max_waits; ++waits) {
                while (body.size() < pos + csize + 2) {
                    if (std::chrono::steady_clock::now() > deadline) {
                        recv_failed = false;
                        break;   // the deadline is the reason
                    }
                    const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
                    if (n > 0) { body.append(buf, size_t(n)); continue; }
                    if (n == 0) { recv_failed = true; break; }   // peer closed
                    if (errno == EINTR) continue;
                    // EAGAIN/EWOULDBLOCK: the read timed out. Wait and
                    // retry -- the server is probably still thinking.
                    break;
                }
                if (body.size() >= pos + csize + 2 || recv_failed) break;
                if (std::chrono::steady_clock::now() > deadline) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
            chunk_complete = (body.size() >= pos + csize + 2);
        }
        if (!chunk_complete) {
            // The chunk never arrived in full. Report it as a REAL error
            // rather than parsing a truncated buffer.
            ::close(fd);
            {
                std::lock_guard<std::mutex> lk(mu_);
                error_ = recv_failed
                    ? "the server closed the connection mid-response"
                    : "the response timed out while reading a chunk";
            }
            state_.store(State::Failed, std::memory_order_release);
            return;
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
        if (final_.empty()) {
            // Name the ACTUAL cause: a deadline points at the CLOCK, not at
            // the model.
            error_ = (std::chrono::steady_clock::now() > deadline)
                ? "the request exceeded its time budget"
                : "the model returned nothing";
        }
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
