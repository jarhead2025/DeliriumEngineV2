// =============================================================================
// validate_escalation_rescue - GATE for the escalation ladder's RESCUE.
//
// The ladder's WIRING is proven by validate_escalation (36 checks): the
// escalated attempt goes to the larger model and only to it, it carries the
// stage/reason/log/worked-examples, and a successful escalation becomes
// corpus.
//
// What was NOT proven: that the escalation actually RESCUES -- that a
// bigger model fixes what the smaller one could not. Every live run so far
// succeeded on attempt 1, so the second rung never fired against a real
// second endpoint.
//
// This gate proves the rescue MECHANIC without needing two live models:
// it supplies two DIFFERENT responders -- a WEAK one that always fails and
// a STRONG one that always succeeds -- and asserts that the ladder
// actually reaches the strong one, and that the answer comes from the
// strong one.
//
// The live two-model run is a separate, LABELLED check (it needs Ollama and
// takes minutes). This gate is the deterministic proof of the mechanic; the
// live run is the proof of the BEHAVIOUR.
//
// The invariant, stated once:
//   "If the local model fails, the ladder MUST consult the larger model,
//    and the larger model's answer MUST be the one that is used."
// =============================================================================
#include "delirium/ai/DocLearner.h"

#include <cstdio>
#include <functional>
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

// A valid two-sheet tree (the document path REJECTS a single sheet).
static const char* kGoodTree = R"({
  "kind": "body", "substance": "ellipsoid",
  "substance_params": {"sx":0.5,"sy":0.5,"sz":0.4},
  "children": [
    {"kind":"rotor", "substance":"ellipsoid",
     "substance_params":{"sx":0.17,"sy":0.17,"sz":0.2},
     "origin":[0,0,0]}
  ]
})";

// A SINGLE sheet: the document path rejects this as "only one face".
// It is what a small model produces for a document it cannot decompose.
static const char* kSingleSheetTree = R"({
  "kind": "body", "substance": "ellipsoid",
  "substance_params": {"sx":0.5,"sy":0.5,"sz":0.5}
})";

static void test_the_ladder_rescues_a_failure() {
    DocErrorLog log;
    Document doc;
    doc.id = "V-300";
    doc.domain = "machine";
    doc.text = "A ball valve with a body, a ball, a stem and a handle.";

    int weak_calls = 0, strong_calls = 0;
    // The WEAK model: always answers with a single sheet, which the
    // document path rejects as "only one face".
    const auto weak = [&](const std::string&, const std::string&) {
        ++weak_calls;
        return std::string(kSingleSheetTree);
    };
    // The STRONG model: answers with a valid multi-sheet tree.
    const auto strong = [&](const std::string&, const std::string&) {
        ++strong_calls;
        return std::string(kGoodTree);
    };

    // Attempt 1: the weak model fails.
    DocResult r1 = InterpretDocument(doc, weak, log);
    CHECK(weak_calls == 1, "attempt 1 must ask the weak model once");
    CHECK(!r1.ok, "attempt 1 must fail");
    CHECK(r1.failed_stage == "validate",
          "the weak model's single sheet must fail at validate: got '%s'",
          r1.failed_stage.c_str());

    // Attempt 2: the escalation.
    const std::function<std::string(const std::string&, const std::string&)>
        strong_fn = strong;
    DocResult r2 = InterpretDocument(doc, weak, log, /*escalate=*/true, &strong_fn);

    CHECK(strong_calls == 1,
          "the escalation MUST consult the larger model: strong_calls=%d",
          strong_calls);
    CHECK(weak_calls == 1,
          "the escalation must NOT re-ask the weak model: weak_calls=%d",
          weak_calls);
    CHECK(r2.ok, "the larger model must RESCUE the failure: %s",
          r2.reason.c_str());
    CHECK(r2.escalated, "the rescued result must be flagged escalated");
    CHECK(r2.tree != nullptr, "the rescued result must own a tree");
}

static void test_the_rescued_answer_is_the_one_used() {
    // The stronger claim: the answer that comes back is the LARGER model's,
    // not the weak model's. A ladder that consults the strong model and
    // then discards its answer has not rescued anything.
    DocErrorLog log;
    Document doc;
    doc.id = "V-400";
    doc.domain = "machine";
    doc.text = "A ball valve.";

    const auto weak = [](const std::string&, const std::string&) {
        return std::string(kSingleSheetTree);
    };
    const auto strong = [](const std::string&, const std::string&) {
        return std::string(kGoodTree);
    };
    const std::function<std::string(const std::string&, const std::string&)>
        strong_fn = strong;

    InterpretDocument(doc, weak, log);
    const DocResult r = InterpretDocument(doc, weak, log, true, &strong_fn);
    CHECK(r.ok, "the escalation must succeed");
    if (r.tree) {
        // The strong model's tree has a CHILD (two sheets); the weak
        // model's does not. So the presence of a child proves which
        // answer was used -- without depending on the exact bytes.
        CHECK(!r.tree->children.empty(),
              "the used tree must be the larger model's (it has children)");
        const std::string& kind = r.tree->kind;
        CHECK(kind == "body", "the root's kind must come from the strong model");
    }
}

static void test_the_escalation_carries_the_failure_to_the_strong_model() {
    // The rescue depends on the strong model being TOLD what went wrong.
    // Without the failure reason, the strong model would be asked the same
    // question and would likely fail the same way.
    DocErrorLog log;
    Document doc;
    doc.id = "V-500";
    doc.domain = "machine";
    doc.text = "A ball valve.";

    std::string escalated_user;
    const auto weak = [](const std::string&, const std::string&) {
        return std::string(kSingleSheetTree);
    };
    const auto strong = [&](const std::string&, const std::string& user) {
        escalated_user = user;
        return std::string(kGoodTree);
    };
    const std::function<std::string(const std::string&, const std::string&)>
        strong_fn = strong;

    InterpretDocument(doc, weak, log);
    InterpretDocument(doc, weak, log, true, &strong_fn);

    CHECK(escalated_user.find("validate") != std::string::npos,
          "the strong model must be told WHICH STAGE failed");
    CHECK(escalated_user.find("sheet") != std::string::npos,
          "the strong model must be told the failure REASON");
    // The DOCUMENT must be restated: the escalation prompt has to be
    // self-contained, because the strong model has never seen this document.
    CHECK(escalated_user.find("ball valve") != std::string::npos,
          "the strong model must be given the DOCUMENT again");
}

static void test_no_rescue_without_a_larger_model() {
    // With no larger model, the ladder re-asks the LOCAL model with the
    // enriched context. If the local model always fails, the escalation
    // fails too -- and must SAY so, rather than reporting success.
    DocErrorLog log;
    Document doc;
    doc.id = "V-600";
    doc.domain = "machine";
    doc.text = "A ball valve.";

    int calls = 0;
    const auto weak = [&](const std::string&, const std::string&) {
        ++calls;
        return std::string(kSingleSheetTree);
    };
    InterpretDocument(doc, weak, log);
    const DocResult r = InterpretDocument(doc, weak, log, true);
    CHECK(calls == 2, "the local retry must ask the local model");
    CHECK(!r.ok, "an unrescuable failure must stay a failure");
    CHECK(r.escalated, "the failed retry must still be flagged escalated");
    CHECK(log.FailureCount() == 2, "both failures must be logged");
}

static void test_the_rescue_becomes_corpus() {
    // The rescued success must be retrievable -- that is what makes the
    // ladder CUMULATIVE rather than per-document.
    DocErrorLog log;
    Document doc;
    doc.id = "V-700";
    doc.domain = "machine";
    doc.text = "A ball valve with a body, a ball, a stem and a handle.";

    const auto weak = [](const std::string&, const std::string&) {
        return std::string(kSingleSheetTree);
    };
    const auto strong = [](const std::string&, const std::string&) {
        return std::string(kGoodTree);
    };
    const std::function<std::string(const std::string&, const std::string&)>
        strong_fn = strong;

    InterpretDocument(doc, weak, log);
    InterpretDocument(doc, weak, log, true, &strong_fn);
    CHECK(log.SuccessCount() == 1, "the rescue must be logged as a success");

    // A later, similar document must find it.
    Document later;
    later.id = "V-800";
    later.domain = "machine";
    later.text = "Another valve with a spherical ball and a stem handle.";
    const auto hits = log.RetrieveSimilar(later.text, later.domain, 3);
    CHECK(!hits.empty(),
          "the rescued success must be retrievable as corpus");
    if (!hits.empty()) {
        CHECK(hits[0].document_id == "V-700",
              "the retrieved example must be the rescued one: got '%s'",
              hits[0].document_id.c_str());
    }
}

int main() {
    test_the_ladder_rescues_a_failure();
    test_the_rescued_answer_is_the_one_used();
    test_the_escalation_carries_the_failure_to_the_strong_model();
    test_no_rescue_without_a_larger_model();
    test_the_rescue_becomes_corpus();

    std::printf("\nvalidate_escalation_rescue: %d checks, %d failures\n",
                g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
