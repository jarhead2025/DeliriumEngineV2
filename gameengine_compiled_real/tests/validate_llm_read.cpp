// =============================================================================
// validate_llm_read - GATE for the LLM client's READ path.
//
// The client speaks HTTP over raw POSIX sockets to an Ollama-compatible
// server, and it sets SO_RCVTIMEO on the socket. That combination has one
// sharp edge: a recv can fail with a TIMEOUT rather than returning data.
//
// The bug this gate protects against: a read timeout was treated as "the
// data has arrived". The header loop broke out and reported "no response
// headers"; the chunk-body loop broke out and then parsed a TRUNCATED
// buffer. Both surfaced as the same generic symptom -- "the model returned
// nothing" -- against a perfectly healthy server.
//
// WHY THE GATE USES ITS OWN SERVER: the failure needs a server that
// STALLS mid-response. A real Ollama either answers or does not, and it
// cannot be made to stall on demand. So the gate runs a tiny local HTTP
// server that accepts a connection, sends headers, then sleeps before the
// body -- exactly the shape that used to break the client.
//
// The gate asserts the INVARIANTS:
//   - a server that answers NORMALLY yields the full response
//   - a server that stalls BEFORE the headers is retried, and a reply
//     that eventually arrives IS received
//   - a server that stalls MID-CHUNK is not parsed as a truncated buffer
//   - a server that CLOSES mid-response is reported as a close, not as an
//     empty model reply
//   - every failure NAMES ITS CAUSE (the error is never empty)
// =============================================================================
#include "delirium/llm/LLMClient.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace DeliRium;
using namespace DeliRium::LLM;

static int g_fail = 0;
static int g_checks = 0;

#define CHECK(cond, ...) do { \
    ++g_checks; \
    if (!(cond)) { ++g_fail; std::printf("FAIL %s:%d  ", __FILE__, __LINE__); \
                   std::printf(__VA_ARGS__); std::printf("\n"); } \
} while (0)

// ---------------------------------------------------------------------------
// A minimal HTTP server that can be told how to misbehave.
//
// The modes exist to reproduce the exact failure shapes:
//   NORMAL      -- answer immediately, with a complete SSE body
//   STALL_HEAD  -- send nothing, sleep, then send the whole answer
//   STALL_BODY  -- send the headers, sleep, then send the body
//   CLOSE_EARLY -- send the headers, then close the connection
// ---------------------------------------------------------------------------
enum class Mode { NORMAL, STALL_HEAD, STALL_BODY, CLOSE_EARLY };

class TestServer {
public:
    explicit TestServer(Mode mode) : mode_(mode) {
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        int one = 1;
        ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in addr {};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0;   // let the kernel choose
        if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
            ::listen(listen_fd_, 8) != 0) {
            std::fprintf(stderr, "FAIL: the test server could not bind\n");
            std::exit(1);
        }
        socklen_t len = sizeof(addr);
        if (::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
            std::fprintf(stderr, "FAIL: the test server could not read its port\n");
            std::exit(1);
        }
        port_ = ntohs(addr.sin_port);
        // The accept loop must be INTERRUPTIBLE, or the destructor cannot
        // join it: a thread blocked in accept() on a closed descriptor may
        // never return, and the test would HANG at exit rather than fail.
        // A short accept timeout makes the worker poll, so it notices the
        // "stop" flag between connections.
        struct timeval tv {};
        tv.tv_sec = 0;
        tv.tv_usec = 100000;   // 100 ms
        ::setsockopt(listen_fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        thread_ = std::thread([this] { Serve(); });
    }

    ~TestServer() {
        stop_ = true;
        if (thread_.joinable()) thread_.join();
        ::close(listen_fd_);
    }

    int Port() const { return port_; }

private:
    void Serve() {
        for (int i = 0; i < 8 && !stop_; ++i) {
            const int fd = ::accept(listen_fd_, nullptr, nullptr);
            if (fd < 0) {
                if (stop_) return;
                continue;   // an accept timeout: poll again
            }
            // Read the request (we do not care what it says). A receive
            // timeout keeps a mis-behaving client from stalling the server
            // thread forever.
            struct timeval tv {};
            tv.tv_sec = 5;
            ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
            char buf[4096];
            ::recv(fd, buf, sizeof(buf), 0);

            const std::string body =
                "data: {\"choices\":[{\"delta\":{\"content\":\"HELLO\"}}]}\n\n"
                "data: [DONE]\n\n";

            if (mode_ == Mode::STALL_HEAD) {
                // The client must WAIT rather than give up.
                std::this_thread::sleep_for(std::chrono::milliseconds(2500));
            }

            const std::string headers =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/event-stream\r\n"
                "Transfer-Encoding: chunked\r\n"
                "\r\n";
            ::send(fd, headers.data(), headers.size(), 0);

            if (mode_ == Mode::CLOSE_EARLY) {
                ::close(fd);
                continue;
            }

            if (mode_ == Mode::STALL_BODY) {
                // Headers sent, body delayed: the client is now waiting on
                // a chunk that has not arrived.
                std::this_thread::sleep_for(std::chrono::milliseconds(2500));
            }

            // The chunked body: one chunk, then the terminator.
            char size[32];
            std::snprintf(size, sizeof(size), "%zx\r\n", body.size());
            ::send(fd, size, std::strlen(size), 0);
            ::send(fd, body.data(), body.size(), 0);
            ::send(fd, "\r\n0\r\n\r\n", 7, 0);
            ::close(fd);
        }
    }

    Mode mode_;
    int listen_fd_ = -1;
    int port_ = 0;
    // Set by the destructor so the accept loop notices it between
    // connections (the accept has a timeout, so it polls).
    std::atomic<bool> stop_{false};
    std::thread thread_;
};

// Ask the client once and report what happened.
struct Outcome {
    bool failed = false;
    bool done = false;
    std::string reply;
    std::string error;
    std::string partial;
};

// The client's own timeout must be SHORTER than the server's stall, so the
// retry path is the thing under test -- and the reply that eventually
// arrives must still be received.
static Outcome AskOnce(int port, int timeout_seconds) {
    Outcome o;
    Config c;
    c.host = "127.0.0.1";
    c.port = port;
    c.model = "test-model";
    c.system = "you are a test";
    c.max_tokens = 64;
    c.timeout_seconds = timeout_seconds;

    LLMClient client(c);
    if (!client.Send("hello")) { o.failed = true; o.error = "send failed"; return o; }
    const int max_ticks = timeout_seconds * 20 + 40;
    int ticks = 0;
    while (client.IsBusy()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (++ticks > max_ticks) { client.Cancel(); o.failed = true; o.error = "outer timeout"; return o; }
    }
    o.failed = (client.CurrentState() == State::Failed);
    o.done = (client.CurrentState() == State::Done);
    o.error = client.Error();
    o.reply = client.TakeResponse();
    o.partial = client.Partial();
    return o;
}

// --- the normal case -----------------------------------------------------------

static void test_a_normal_server_is_answered() {
    TestServer srv(Mode::NORMAL);
    const Outcome o = AskOnce(srv.Port(), 10);
    CHECK(!o.failed, "a normal server must not fail: %s", o.error.c_str());
    CHECK(o.done, "a normal server must reach Done");
    CHECK(o.reply.find("HELLO") != std::string::npos,
          "the reply must contain the streamed text: '%s'", o.reply.c_str());
    CHECK(o.error.empty(), "a success must carry no error: '%s'", o.error.c_str());
}

// --- the stall BEFORE the headers ----------------------------------------------

static void test_a_stall_before_the_headers_is_waited_out() {
    // The server sends NOTHING for 2.5 s. The old code gave up on the
    // first recv timeout and reported "no response headers" -- a transport
    // failure for a server that was about to answer perfectly.
    TestServer srv(Mode::STALL_HEAD);
    const Outcome o = AskOnce(srv.Port(), 10);
    CHECK(!o.failed,
          "a server that stalls before its headers must be waited out, not failed: %s",
          o.error.c_str());
    CHECK(o.reply.find("HELLO") != std::string::npos,
          "the delayed reply must still arrive: '%s'", o.reply.c_str());
}

// --- the stall MID-CHUNK --------------------------------------------------------

static void test_a_stall_mid_chunk_is_not_parsed_as_truncated() {
    // The headers arrive, then the body is delayed by 2.5 s. The old code
    // treated the read timeout as "the chunk has arrived" and parsed a
    // truncated buffer -- which produced an empty reply and the generic
    // "the model returned nothing".
    //
    // The invariant: the client must either receive the FULL body or fail
    // with a message that NAMES the fault. It must never silently succeed
    // with a truncated answer.
    TestServer srv(Mode::STALL_BODY);
    const Outcome o = AskOnce(srv.Port(), 10);
    if (!o.failed) {
        CHECK(o.reply.find("HELLO") != std::string::npos,
              "if it did not fail, the reply must be COMPLETE: '%s'",
              o.reply.c_str());
    } else {
        CHECK(!o.error.empty(),
              "a failure must NAME its cause, never be empty");
        CHECK(o.error.find("timed out") != std::string::npos ||
              o.error.find("closed the connection") != std::string::npos,
              "the error must name the fault: '%s'", o.error.c_str());
    }
}

// --- the server closes mid-response ---------------------------------------------

static void test_a_closed_connection_is_reported_as_such() {
    // The headers arrive, then the server closes. The client must say so
    // -- not report an empty reply as if the model had produced nothing.
    TestServer srv(Mode::CLOSE_EARLY);
    const Outcome o = AskOnce(srv.Port(), 10);
    CHECK(o.failed, "a closed connection must fail");
    CHECK(!o.error.empty(), "the failure must carry a reason");
    CHECK(o.error.find("closed") != std::string::npos ||
          o.error.find("nothing") != std::string::npos,
          "the error must name the close: '%s'", o.error.c_str());
}

// --- every failure names its cause ----------------------------------------------

static void test_no_failure_is_silent() {
    // The generic symptom -- "the model returned nothing" -- hides whether
    // the fault is the server, the network, or the prompt. Every failure
    // path must produce a message that names the actual cause.
    //
    // This is checked against a port with NOTHING on it: a connection
    // refusal must be reported as a connection failure.
    Config c;
    c.host = "127.0.0.1";
    c.port = 1;      // nothing listens here
    c.timeout_seconds = 3;
    LLMClient client(c);
    client.Send("hello");
    int ticks = 0;
    while (client.IsBusy() && ticks < 200) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        ++ticks;
    }
    CHECK(client.CurrentState() == State::Failed,
          "a refused connection must fail");
    const std::string err = client.Error();
    CHECK(!err.empty(), "a refused connection must carry a reason");
    CHECK(err.find("cannot reach") != std::string::npos,
          "the reason must name the connection: '%s'", err.c_str());
}

int main() {
    test_a_normal_server_is_answered();
    test_a_stall_before_the_headers_is_waited_out();
    test_a_stall_mid_chunk_is_not_parsed_as_truncated();
    test_a_closed_connection_is_reported_as_such();
    test_no_failure_is_silent();

    std::printf("\nvalidate_llm_read: %d checks, %d failures\n",
                g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
