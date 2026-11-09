// =============================================================================
// validate_doc_learner - GATE for the document-driven reconstruction loop.
//
// The learner turns a TECHNICAL DOCUMENT into an element tree, logs every
// failure with its reason, and escalates to a larger model when the local
// one cannot cope.
//
// The gate asserts the INVARIANTS of that loop (not the specific values):
//   - a valid multi-part document becomes a multi-leaf (closed-body) tree
//   - a single-leaf answer is REJECTED as "only one sheet", because that
//     is the exact "the sphere only shows one face" failure
//   - every failure is LOGGED with a stage and a reason
//   - the error log carries the failures FORWARD into the escalation
//     prompt, so the second attempt sees what the first one got wrong
//   - the escalation goes to the LARGER model when one is supplied
//   - the log survives a save/load round-trip
//   - a transport failure is logged as such, not as a parse failure
// =============================================================================
#include "delirium/ai/DocLearner.h"
#include "delirium/ai/ElementTreeParser.h"
#include "delirium/ai/ElementSystem.h"
#include "delirium/psf/PSFBody.h"

#include <cctype>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace DeliRium;
using namespace DeliRium::AI;

static int g_fail = 0;
static int g_checks = 0;

#define CHECK(cond, ...) do { \
    ++g_checks; \
    if (!(cond)) { ++g_fail; std::printf("FAIL %s:%d  ", __FILE__, __LINE__); \
                   std::printf(__VA_ARGS__); std::printf("\n"); } \
} while (0)

// A two-part machine document: a housing with a rotor inside it.
static const char* kMachineDoc =
    "Assembly A-100: a cylindrical housing, 120mm diameter, 80mm tall, with "
    "a rotor of 40mm diameter concentrically mounted inside it. The rotor "
    "sits at the housing's centre and is free to rotate.";

static const char* kTwoLeafTree = R"({
  "kind": "body", "substance": "ellipsoid",
  "substance_params": {"sx":0.5,"sy":0.5,"sz":0.4},
  "children": [
    {"kind":"rotor", "substance":"ellipsoid",
     "substance_params":{"sx":0.17,"sy":0.17,"sz":0.2},
     "origin":[0,0,0]}
  ]
})";

static const char* kOneLeafTree = R"({
  "kind": "body", "substance": "ellipsoid",
  "substance_params": {"sx":0.5,"sy":0.5,"sz":0.5}
})";

// --- the learner --------------------------------------------------------------

static void test_closed_body_rule() {
    DocErrorLog log;
    Document doc;
    doc.id = "A-100";
    doc.domain = "machine";
    doc.text = kMachineDoc;

    // A model that answers with a single sheet must be REJECTED: that is
    // the "only one face" failure.
    {
        const auto ask = [](const std::string&, const std::string&) {
            return std::string(kOneLeafTree);
        };
        DocResult r = InterpretDocument(doc, ask, log);
        CHECK(!r.ok, "a single-leaf tree must be rejected");
        CHECK(r.failed_stage == "validate",
              "expected stage 'validate', got '%s'", r.failed_stage.c_str());
        CHECK(r.reason.find("ONE sheet") != std::string::npos,
              "the reason must name the single-sheet problem: %s", r.reason.c_str());
        CHECK(log.FailureCount() == 1, "the failure must be logged");
    }

    // A multi-part tree must be ACCEPTED and bake into a real body.
    {
        const auto ask = [](const std::string&, const std::string&) {
            return std::string(kTwoLeafTree);
        };
        DocResult r = InterpretDocument(doc, ask, log);
        CHECK(r.ok, "a two-leaf tree must be accepted: %s", r.reason.c_str());
        CHECK(r.tree != nullptr, "an accepted result must own a tree");
        CHECK(log.SuccessCount() == 1, "the success must be logged");

        if (r.tree) {
            // AND the tree must BAKE into a real body -- the seam is
            // "a document becomes geometry", not "a document becomes JSON".
            std::unique_ptr<PSFBody> body;
            std::string why;
            body = ElementSystem::BakeBody(*r.tree, 24, 1.0f, why);
            CHECK(body != nullptr, "the tree must bake into a body: %s", why.c_str());
            if (body) {
                CHECK(body->Patches().size() > 0, "the baked body must have patches");
                CHECK(body->AnchorsConst().Count() > 0,
                      "the baked body must have anchors");
            }
        }
    }
}

static void test_error_log_feeds_escalation() {
    DocErrorLog log;
    Document doc;
    doc.id = "A-100";
    doc.text = kMachineDoc;

    // Attempt 1: the local model produces a single sheet -> rejected.
    std::string seen_user_2;
    {
        const auto ask = [](const std::string&, const std::string&) {
            return std::string(kOneLeafTree);
        };
        DocResult r1 = InterpretDocument(doc, ask, log);
        CHECK(!r1.ok, "attempt 1 must fail");
        CHECK(r1.failed_stage == "validate", "attempt 1 stage");
    }

    // Attempt 2 (escalated): the prompt must CARRY the previous failure.
    {
        const auto ask = [&](const std::string&, const std::string& user) {
            seen_user_2 = user;
            return std::string(kTwoLeafTree);
        };
        DocResult r2 = InterpretDocument(doc, ask, log, /*escalate=*/true);
        CHECK(r2.ok, "the escalated attempt must succeed: %s", r2.reason.c_str());
        CHECK(r2.escalated, "the result must be marked escalated");
        CHECK(seen_user_2.find("FAILED") != std::string::npos,
              "the escalation prompt must include the failure log");
        // The escalation prompt must name the PREVIOUS failure's reason,
        // not merely say "FAILED" -- the reason is what tells the model
        // what to change. The check is case-insensitive because the
        // reason's exact wording is an implementation detail; what
        // matters is that the CONTENT reaches the prompt.
        {
            const std::string up = [&]{ std::string s = seen_user_2;
                for (char& ch : s) ch = (char)std::toupper((unsigned char)ch);
                return s; }();
            CHECK(up.find("ONE SHEET") != std::string::npos,
                  "the escalation prompt must name the previous failure "
                  "(single-sheet); prompt was:\n%s", seen_user_2.c_str());
        }
        CHECK(seen_user_2.find(kMachineDoc) != std::string::npos,
              "the escalation prompt must still carry the document");
    }
}

static void test_larger_model_escalation() {
    DocErrorLog log;
    Document doc;
    doc.id = "B-200";
    doc.text = "a sphere of 100mm diameter";

    bool larger_called = false;
    bool local_called = false;

    const auto local = [&](const std::string&, const std::string&) {
        local_called = true;
        return std::string(kOneLeafTree);   // always wrong
    };
    const auto larger = [&](const std::string&, const std::string&) {
        larger_called = true;
        return std::string(kTwoLeafTree);   // correct
    };

    // Attempt 1 uses the local model and fails.
    DocResult r1 = InterpretDocument(doc, local, log);
    CHECK(local_called, "attempt 1 must use the local model");
    CHECK(!r1.ok, "attempt 1 must fail");

    // Attempt 2 escalates: the LARGER model must be consulted, and the
    // local one must NOT be.
    local_called = false;
    // The larger-model callback is taken as a POINTER to a std::function,
    // so it must be wrapped (a bare lambda does not implicitly convert to
    // a std::function temporary that outlives the call).
    const std::function<std::string(const std::string&, const std::string&)>
        larger_fn = larger;
    DocResult r2 = InterpretDocument(doc, local, log, /*escalate=*/true, &larger_fn);
    CHECK(larger_called, "the escalation must consult the larger model");
    CHECK(!local_called, "the escalation must NOT re-ask the local model");
    CHECK(r2.ok, "the larger model must succeed: %s", r2.reason.c_str());
    CHECK(r2.escalated, "the result must be marked escalated");
    CHECK(log.EscalatedCount() == 1, "the escalation must be logged as such");
}

static void test_transport_failure() {
    DocErrorLog log;
    Document doc;
    doc.id = "C-300";
    doc.text = "whatever";

    const auto ask = [](const std::string&, const std::string&) {
        return std::string();   // unreachable
    };
    DocResult r = InterpretDocument(doc, ask, log);
    CHECK(!r.ok, "a transport failure must fail");
    CHECK(r.failed_stage == "transport",
          "expected stage 'transport', got '%s'", r.failed_stage.c_str());
    CHECK(log.FailureCount() == 1, "the transport failure must be logged");
    CHECK(log.Attempts().back().failed_stage == "transport",
          "the log must record the transport stage");
}

static void test_parse_failure() {
    DocErrorLog log;
    Document doc;
    doc.id = "D-400";
    doc.text = "whatever";

    const auto ask = [](const std::string&, const std::string&) {
        return std::string("this is not JSON at all");
    };
    DocResult r = InterpretDocument(doc, ask, log);
    CHECK(!r.ok, "a non-JSON reply must fail");
    CHECK(r.failed_stage == "parse",
          "expected stage 'parse', got '%s'", r.failed_stage.c_str());
    CHECK(!r.reason.empty(), "a parse failure must carry a reason");
}

static void test_log_roundtrip() {
    DocErrorLog log;
    Document doc;
    doc.id = "E-500";
    doc.text = "a thing";

    const auto bad = [](const std::string&, const std::string&) {
        return std::string(kOneLeafTree);
    };
    const auto good = [](const std::string&, const std::string&) {
        return std::string(kTwoLeafTree);
    };
    InterpretDocument(doc, bad, log);
    InterpretDocument(doc, good, log);

    const std::string path = "/tmp/hermes-doc-learner-log.txt";
    CHECK(log.Save(path), "the log must save");
    CHECK(log.FailureCount() == 1, "one failure before the round-trip");
    CHECK(log.SuccessCount() == 1, "one success before the round-trip");

    DocErrorLog reloaded;
    CHECK(reloaded.Load(path), "the log must load");
    CHECK(reloaded.Size() == log.Size(), "the entry count must survive");
    CHECK(reloaded.FailureCount() == log.FailureCount(), "the failures must survive");
    CHECK(reloaded.SuccessCount() == log.SuccessCount(), "the successes must survive");
    CHECK(reloaded.Attempts().back().ok, "the last entry's status must survive");
    CHECK(reloaded.Attempts().back().document_id == "E-500",
          "the document id must survive");

    // The reloaded log must still be usable as prompt context.
    const std::string ctx = reloaded.AsPromptContext();
    CHECK(ctx.find("E-500") != std::string::npos,
          "the reloaded log must still produce prompt context");
}

static void test_prompt_contract() {
    const std::string p = DocToTreeSystemPrompt();
    // The prompt IS the schema: every legal leaf must appear in it.
    CHECK(p.find("ellipsoid") != std::string::npos, "the prompt must list ellipsoid");
    CHECK(p.find("torus") != std::string::npos, "the prompt must list torus");
    CHECK(p.find("plane") != std::string::npos, "the prompt must list plane");
    CHECK(p.find("heightfield") != std::string::npos, "the prompt must list heightfield");
    // The CLOSED-BODY RULE must be stated -- it is the rule that stops the
    // "only one face" output.
    CHECK(p.find("CLOSED-BODY") != std::string::npos,
          "the prompt must state the closed-body rule");
    // It must instruct the model to emit only JSON.
    CHECK(p.find("ONLY the JSON") != std::string::npos,
          "the prompt must demand JSON-only output");
}

int main() {
    test_closed_body_rule();
    test_error_log_feeds_escalation();
    test_larger_model_escalation();
    test_transport_failure();
    test_parse_failure();
    test_log_roundtrip();
    test_prompt_contract();

    std::printf("\nvalidate_doc_learner: %d checks, %d failures\n",
                g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
