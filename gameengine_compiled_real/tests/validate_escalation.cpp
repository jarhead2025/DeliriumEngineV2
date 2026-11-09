// =============================================================================
// validate_escalation - GATE for the escalation ladder (G14).
//
// The learner's ladder:
//   attempt 1: the local model, with the plain prompt.
//   attempt 2: a LARGER model, with the failure reason + the accumulated
//              error log + the retrieved worked examples attached.
//
// The ladder was unit-tested against a stub and had never been driven
// against a second endpoint. This gate asserts the CONTRACT of the ladder
// -- the properties that make it a real escalation rather than a second
// attempt with different text:
//
//   - the escalated attempt goes to the LARGER model, and ONLY to it
//   - the escalated attempt CARRIES the specific failure reason
//   - the escalated attempt CARRIES the error log (the pattern of what
//     has already failed)
//   - the escalated attempt CARRIES the worked examples
//   - the escalation prompt is SELF-CONTAINED: it must restate the
//     document, because a retry that has to rely on the caller's memory
//     of the previous turn is not a retry
//   - the escalation is RECORDED as escalated
//   - a successful escalation is logged as a SUCCESS (it becomes corpus)
//   - a failed escalation does not erase the original failure
//
// The live-model run (against a real second endpoint) is a separate
// LABELLED gate, because it needs a server and takes minutes. This gate
// proves the WIRING; the labelled gate proves the BEHAVIOUR.
// =============================================================================
#include "delirium/ai/DocLearner.h"
#include "delirium/ai/ElementTreeParser.h"

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

// --- the escalation goes to the larger model, and only to it -------------------

static void test_escalation_goes_to_the_larger_model() {
    DocErrorLog log;
    Document doc;
    doc.id = "A-100";
    doc.domain = "machine";
    doc.text = "A gearbox housing with a rotor inside it.";

    int local_calls = 0, larger_calls = 0;
    const auto local = [&](const std::string&, const std::string&) {
        ++local_calls;
        return std::string(kOneLeafTree);   // always fails: one sheet
    };
    const auto larger = [&](const std::string&, const std::string&) {
        ++larger_calls;
        return std::string(kTwoLeafTree);  // succeeds
    };

    // Attempt 1: the local model fails.
    DocResult r1 = InterpretDocument(doc, local, log);
    CHECK(local_calls == 1, "attempt 1 must ask the local model exactly once");
    CHECK(!r1.ok, "attempt 1 must fail");
    CHECK(r1.failed_stage == "validate",
          "attempt 1 must fail at validate: got '%s'", r1.failed_stage.c_str());

    // Attempt 2: the escalation.
    const std::function<std::string(const std::string&, const std::string&)>
        larger_fn = larger;
    DocResult r2 = InterpretDocument(doc, local, log, /*escalate=*/true, &larger_fn);

    CHECK(larger_calls == 1, "the escalation must ask the larger model once");
    CHECK(local_calls == 1,
          "the escalation must NOT re-ask the local model (local_calls=%d)",
          local_calls);
    CHECK(r2.ok, "the escalation must succeed: %s", r2.reason.c_str());
    CHECK(r2.escalated, "the result must be flagged as escalated");
    CHECK(r2.tree != nullptr, "an escalated success must own a tree");
}

// --- the escalation carries the CONTEXT ----------------------------------------

static void test_escalation_carries_the_failure_reason() {
    DocErrorLog log;
    Document doc;
    doc.id = "A-100";
    doc.domain = "machine";
    doc.text = "A gearbox housing with a rotor inside it.";

    std::string escalated_user;
    const auto local = [](const std::string&, const std::string&) {
        return std::string(kOneLeafTree);
    };
    const auto larger = [&](const std::string&, const std::string& user) {
        escalated_user = user;
        return std::string(kTwoLeafTree);
    };

    InterpretDocument(doc, local, log);
    const std::function<std::string(const std::string&, const std::string&)>
        larger_fn = larger;
    InterpretDocument(doc, local, log, /*escalate=*/true, &larger_fn);

    // The SPECIFIC failure reason must be present. The reason for the
    // one-sheet failure mentions "one sheet" -- but the exact wording is
    // the implementation's business, so the check is on the STAGE, which
    // is the stable part of the contract.
    CHECK(escalated_user.find("validate") != std::string::npos,
          "the escalation prompt must name the failed stage");
    CHECK(escalated_user.find("sheet") != std::string::npos,
          "the escalation prompt must carry the failure reason");
    // The escalation prompt must RESTATE the document: a retry that has
    // to rely on the caller's memory of the previous turn is not a retry.
    CHECK(escalated_user.find("rotor") != std::string::npos,
          "the escalation prompt must restate the document");
    // It must carry the error log.
    CHECK(escalated_user.find("PREVIOUS FAILURES") != std::string::npos,
          "the escalation prompt must carry the error log");
    // There are NO successes in this log (the only attempt failed), so there
    // are no worked examples to carry -- and the prompt must NOT contain an
    // EMPTY worked-examples section. A header with nothing under it reads as
    // "the examples follow" and wastes the model's attention on a section
    // that does not exist.
    //
    // (An earlier version of this test asserted the opposite -- that the
    // escalation ALWAYS carries worked examples -- which is false for a log
    // with no successes. The assertion belonged in the next test, which
    // builds a corpus that HAS one.)
    CHECK(escalated_user.find("WORKED EXAMPLES") == std::string::npos,
          "with no successes the escalation must NOT carry an empty "
          "worked-examples section");
}

static void test_escalation_carries_a_prior_success_as_a_worked_example() {
    // A corpus with one SUCCESS on a similar document. The escalation for
    // a NEW document must receive that success as a worked example -- it
    // is the payload the larger model needs.
    DocErrorLog log;

    Document good; good.id = "A-000"; good.domain = "machine";
    good.text = "A gearbox housing with a rotor inside it.";
    const auto ok_ask = [](const std::string&, const std::string&) {
        return std::string(kTwoLeafTree);
    };
    InterpretDocument(good, ok_ask, log);
    CHECK(log.SuccessCount() == 1, "the corpus must hold one success");

    Document doc; doc.id = "A-200"; doc.domain = "machine";
    doc.text = "A second gearbox housing with a driven gear inside it.";

    std::string escalated_user;
    const auto local = [](const std::string&, const std::string&) {
        return std::string(kOneLeafTree);
    };
    const auto larger = [&](const std::string&, const std::string& user) {
        escalated_user = user;
        return std::string(kTwoLeafTree);
    };
    InterpretDocument(doc, local, log);
    const std::function<std::string(const std::string&, const std::string&)>
        larger_fn = larger;
    InterpretDocument(doc, local, log, /*escalate=*/true, &larger_fn);

    CHECK(escalated_user.find("WORKED EXAMPLES") != std::string::npos,
          "the escalation must carry the worked examples");
    CHECK(escalated_user.find(kTwoLeafTree) != std::string::npos,
          "the escalation must carry the prior ACCEPTED TREE, not just its id");
}

// --- the ladder is recorded -----------------------------------------------------

static void test_escalation_is_recorded() {
    DocErrorLog log;
    Document doc;
    doc.id = "A-100";
    doc.domain = "machine";
    doc.text = "A gearbox housing with a rotor inside it.";

    const auto local = [](const std::string&, const std::string&) {
        return std::string(kOneLeafTree);
    };
    const auto larger = [](const std::string&, const std::string&) {
        return std::string(kTwoLeafTree);
    };

    InterpretDocument(doc, local, log);
    const std::function<std::string(const std::string&, const std::string&)>
        larger_fn = larger;
    InterpretDocument(doc, local, log, /*escalate=*/true, &larger_fn);

    CHECK(log.Size() == 2, "both attempts must be logged: got %zu", log.Size());
    CHECK(log.FailureCount() == 1, "the original failure must remain logged");
    CHECK(log.SuccessCount() == 1, "the escalated success must be logged");
    CHECK(log.EscalatedCount() == 1, "exactly one attempt must be flagged escalated");

    // The successful escalation becomes CORPUS: a later document must be
    // able to retrieve it. This is the property that makes the ladder
    // cumulative rather than per-document.
    const auto hits = log.RetrieveSimilar(
        "A third gearbox with a housing and a rotor inside it.", "machine", 3);
    CHECK(!hits.empty(), "the escalated success must be retrievable as corpus");
    if (!hits.empty()) {
        CHECK(hits[0].document_id == "A-100",
              "the escalated success must be the retrieved example: got '%s'",
              hits[0].document_id.c_str());
    }
}

static void test_a_failed_escalation_keeps_the_original_failure() {
    // If the escalation ALSO fails, the log must hold BOTH failures --
    // the original and the retry. Losing the original would erase the
    // evidence of what the local model got wrong.
    DocErrorLog log;
    Document doc;
    doc.id = "A-100";
    doc.domain = "machine";
    doc.text = "A gearbox housing with a rotor inside it.";

    const auto local = [](const std::string&, const std::string&) {
        return std::string(kOneLeafTree);
    };
    const auto larger = [](const std::string&, const std::string&) {
        return std::string("still not json");   // fails at parse
    };

    InterpretDocument(doc, local, log);
    const std::function<std::string(const std::string&, const std::string&)>
        larger_fn = larger;
    DocResult r2 = InterpretDocument(doc, local, log, /*escalate=*/true, &larger_fn);

    CHECK(!r2.ok, "the escalation must fail too");
    CHECK(r2.failed_stage == "parse",
          "the escalation must fail at parse: got '%s'", r2.failed_stage.c_str());
    CHECK(r2.escalated, "the failed escalation must still be flagged escalated");
    CHECK(log.FailureCount() == 2,
          "both failures must be logged: got %zu", log.FailureCount());
    CHECK(log.EscalatedCount() == 1, "one attempt must be flagged escalated");
}

static void test_escalation_without_a_larger_model_still_works() {
    // With no larger model supplied, the escalation re-asks the LOCAL model
    // with the enriched context. That is a legitimate (if weaker) ladder
    // rung: the context is the same, only the model is the same too.
    DocErrorLog log;
    Document doc;
    doc.id = "A-100";
    doc.domain = "machine";
    doc.text = "A gearbox housing with a rotor inside it.";

    int calls = 0;
    std::string escalated_user;
    const auto ask = [&](const std::string&, const std::string& user) {
        ++calls;
        if (calls == 1) return std::string(kOneLeafTree);   // fail
        escalated_user = user;
        return std::string(kTwoLeafTree);                   // succeed
    };

    InterpretDocument(doc, ask, log);
    DocResult r2 = InterpretDocument(doc, ask, log, /*escalate=*/true);

    CHECK(calls == 2, "the escalation must re-ask the local model");
    CHECK(r2.ok, "the enriched retry must succeed: %s", r2.reason.c_str());
    CHECK(r2.escalated, "the result must be flagged escalated");
    CHECK(escalated_user.find("PREVIOUS FAILURES") != std::string::npos,
          "the local retry must also carry the error log");
}

static void test_escalation_prompt_is_self_contained() {
    // The escalation prompt is built by a FREE FUNCTION, so it can be
    // checked directly -- and it must be usable on its own, without any
    // state from the caller.
    const std::string p = DocToTreeEscalationPrompt(
        "A gearbox with a rotor.",
        "stage=validate: only one sheet",
        "FAILED document=X stage=parse: bad json",
        "OK document=Y:\n{\"kind\":\"body\"}");

    CHECK(p.find("A gearbox with a rotor.") != std::string::npos,
          "the escalation prompt must carry the document");
    CHECK(p.find("only one sheet") != std::string::npos,
          "the escalation prompt must carry the prior failure");
    CHECK(p.find("bad json") != std::string::npos,
          "the escalation prompt must carry the error log");
    CHECK(p.find("{\"kind\":\"body\"}") != std::string::npos,
          "the escalation prompt must carry the worked example");
    CHECK(p.find("ONLY the JSON") != std::string::npos,
          "the escalation prompt must still demand JSON-only output");
}

int main() {
    test_escalation_goes_to_the_larger_model();
    test_escalation_carries_the_failure_reason();
    test_escalation_carries_a_prior_success_as_a_worked_example();
    test_escalation_is_recorded();
    test_a_failed_escalation_keeps_the_original_failure();
    test_escalation_without_a_larger_model_still_works();
    test_escalation_prompt_is_self_contained();

    std::printf("\nvalidate_escalation: %d checks, %d failures\n",
                g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
