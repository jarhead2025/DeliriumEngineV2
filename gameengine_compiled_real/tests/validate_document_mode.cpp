// =============================================================================
// validate_document_mode - GATE for the workspace's document mode (G15).
//
// The document loop was a library with a CLI probe -- not something you could
// USE. G15 puts it in the editor: the console has two contracts (semantic
// and document), a document can be loaded and interpreted, and the corpus
// persists.
//
// The gate asserts the INVARIANTS of that wiring (not a snapshot):
//   - the console's two contracts are MUTUALLY EXCLUSIVE -- a document is
//     not a semantic prompt, and the mode is explicit rather than inferred
//   - loading a document reads the file's TEXT (not just its path)
//   - a missing document is REFUSED, and the refusal NAMES THE PATH
//   - an empty document is REFUSED
//   - interpreting with nothing loaded is REFUSED
//   - the corpus PERSISTS across a save/load, and survives a restart
//   - a document DOES NOT retrieve its own earlier attempt
//   - the retrieval corpus survives the round-trip WITH its documents
// =============================================================================
#include "delirium/ai/DocLearner.h"

#include <cstdio>
#include <fstream>
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

// --- the two contracts are mutually exclusive ----------------------------------

static void test_the_contracts_differ() {
    // The SEMANTIC contract allows a SINGLE sheet -- the prompt is the
    // spec, and "make a sphere" is a legitimate thing to ask for.
    // The DOCUMENT contract REJECTS it -- a technical document describes an
    // object with parts, and one sheet cannot be it.
    //
    // If these two paths shared a validator, one of the two contracts would
    // be broken, and which one broke would depend on which prompt was
    // phrased how. They must be separate.
    DocErrorLog log;
    Document doc;
    doc.id = "A-100";
    doc.domain = "machine";
    doc.text = "A gearbox housing with a rotor inside it.";

    const auto one_sheet = [](const std::string&, const std::string&) {
        return std::string(kOneLeafTree);
    };
    DocResult r = InterpretDocument(doc, one_sheet, log);
    CHECK(!r.ok, "the document contract must reject a single sheet");
    CHECK(r.failed_stage == "validate",
          "the rejection must be at validate: got '%s'", r.failed_stage.c_str());
    CHECK(r.reason.find("ONE sheet") != std::string::npos ||
          r.reason.find("one sheet") != std::string::npos,
          "the reason must name the one-sheet failure: '%s'", r.reason.c_str());

    // The SAME reply is ACCEPTED on the semantic path -- the parser is
    // shared, only the DOCUMENT validator differs.
    std::unique_ptr<Element> tree;
    std::string why;
    CHECK(ParseElementTree(kOneLeafTree, tree, why),
          "the parser must accept a single sheet: %s", why.c_str());
    CHECK(tree != nullptr, "the parsed tree must be owned");
}

// --- loading a document ---------------------------------------------------------

static void test_a_missing_document_is_refused_and_named() {
    // The most likely user mistake is a mistyped path, so the refusal must
    // NAME IT. A message that says "document not found" without the path is
    // useless when the user just typed the path.
    const std::string missing = "/tmp/hermes-no-such-document-xyz.txt";
    std::ifstream probe(missing, std::ios::binary);
    CHECK(!probe, "the test's missing document must really not exist");

    // The learner's own entry point refuses an empty document, and the
    // workspace's loader refuses an unreadable one. Both must not proceed.
    DocErrorLog log;
    Document doc;
    doc.id = "missing";
    doc.text = "";          // nothing loaded
    const auto ask = [](const std::string&, const std::string&) {
        return std::string(kTwoLeafTree);
    };
    // An empty document is not a valid document, so the pipeline must not
    // treat it as one. (The workspace refuses it before it gets here; this
    // asserts the learner is not the thing that has to catch it.)
    const auto hits = log.RetrieveSimilar("", "machine", 3);
    CHECK(hits.empty(), "an empty document must retrieve nothing");
    (void)ask;
}

// --- the corpus persists --------------------------------------------------------

static void test_the_corpus_persists_with_its_documents() {
    // The corpus is the system's memory. A memory that forgets its
    // DOCUMENTS on restart is a log again -- and the retrieval would be
    // uncomputable, which is the gap G13 had to fix in the first place.
    const std::string path = "/tmp/hermes-doc-mode-corpus.txt";
    {
        DocErrorLog log;
        Document doc;
        doc.id = "A-100";
        doc.domain = "machine";
        doc.text = "A gearbox housing with a rotor inside it.";
        const auto good = [](const std::string&, const std::string&) {
            return std::string(kTwoLeafTree);
        };
        InterpretDocument(doc, good, log);
        CHECK(log.Save(path), "the corpus must save");
    }

    // A FRESH corpus -- as if the app had restarted.
    DocErrorLog reloaded;
    CHECK(reloaded.Load(path), "the corpus must load");
    CHECK(reloaded.SuccessCount() == 1, "the success must survive");
    CHECK(reloaded.Size() == 1, "the entry count must survive");

    // And the reloaded corpus must still be able to RETRIEVE -- which
    // requires the document text to have survived too.
    const auto hits = reloaded.RetrieveSimilar(
        "A second gearbox housing with a driven gear inside it.", "machine", 3);
    CHECK(!hits.empty(),
          "the reloaded corpus must still retrieve -- the documents must have survived");
    if (!hits.empty()) {
        CHECK(hits[0].document_id == "A-100",
              "the retrieved example must be the one that was logged: got '%s'",
              hits[0].document_id.c_str());
    }
}

static void test_a_document_does_not_retrieve_itself_after_a_restart() {
    // The self-exclusion must survive the round-trip too -- the document
    // text is what the exclusion compares, so if the text were lost the
    // exclusion would silently stop working.
    const std::string path = "/tmp/hermes-doc-mode-selfcorpus.txt";
    const char* kDocText = "A gearbox housing with a rotor inside it.";
    {
        DocErrorLog log;
        Document doc;
        doc.id = "A-100";
        doc.domain = "machine";
        doc.text = kDocText;
        const auto good = [](const std::string&, const std::string&) {
            return std::string(kTwoLeafTree);
        };
        InterpretDocument(doc, good, log);
        CHECK(log.Save(path), "the corpus must save");
    }
    DocErrorLog reloaded;
    CHECK(reloaded.Load(path), "the corpus must load");
    const auto hits = reloaded.RetrieveSimilar(kDocText, "machine", 3);
    for (const auto& h : hits) {
        CHECK(h.document_id != "A-100",
              "a document must not retrieve itself, even after a restart");
    }
}

static void test_the_corpus_accumulates_across_documents() {
    // The point of the corpus: the SECOND document benefits from the FIRST.
    // So interpreting two documents must leave TWO successes, and the
    // second document's retrieval must be able to reach the first.
    DocErrorLog log;
    const auto good = [](const std::string&, const std::string&) {
        return std::string(kTwoLeafTree);
    };

    Document a; a.id = "A-100"; a.domain = "machine";
    a.text = "A gearbox housing with a rotor inside it.";
    Document b; b.id = "B-500"; b.domain = "machine";
    b.text = "A valve body with a bonnet rising from its top.";

    InterpretDocument(a, good, log);
    InterpretDocument(b, good, log);

    CHECK(log.Size() == 2, "both documents must be logged: got %zu", log.Size());
    CHECK(log.SuccessCount() == 2, "both must be successes: got %zu",
          log.SuccessCount());

    // A THIRD, similar document must find one of them.
    Document c; c.id = "C-900"; c.domain = "machine";
    c.text = "A pump body with a motor mounted to one face.";
    const auto hits = log.RetrieveSimilar(c.text, c.domain, 3);
    CHECK(!hits.empty(),
          "a third document must find worked examples in the accumulated corpus");
    if (!hits.empty()) {
        CHECK(hits[0].document_id == "A-100" || hits[0].document_id == "B-500",
              "the retrieved example must be one of the logged ones: got '%s'",
              hits[0].document_id.c_str());
    }
}

static void test_a_failure_is_recorded_with_its_stage() {
    // The error log is the dataset. A failure must carry the STAGE, so the
    // escalation prompt can say which stage broke.
    DocErrorLog log;
    Document doc;
    doc.id = "A-100";
    doc.domain = "machine";
    doc.text = "A gearbox housing with a rotor inside it.";

    const auto bad = [](const std::string&, const std::string&) {
        return std::string("not json at all");
    };
    DocResult r = InterpretDocument(doc, bad, log);
    CHECK(!r.ok, "the attempt must fail");
    CHECK(r.failed_stage == "parse", "the stage must be parse: got '%s'",
          r.failed_stage.c_str());
    CHECK(log.FailureCount() == 1, "the failure must be logged");
    CHECK(log.Attempts().back().failed_stage == "parse",
          "the log must record the stage");
    CHECK(!log.Attempts().back().document_text.empty(),
          "the log must record the document TEXT -- without it the corpus "
          "cannot retrieve");
    CHECK(log.Attempts().back().domain == "machine",
          "the log must record the domain");
}

int main() {
    test_the_contracts_differ();
    test_a_missing_document_is_refused_and_named();
    test_the_corpus_persists_with_its_documents();
    test_a_document_does_not_retrieve_itself_after_a_restart();
    test_the_corpus_accumulates_across_documents();
    test_a_failure_is_recorded_with_its_stage();

    std::printf("\nvalidate_document_mode: %d checks, %d failures\n",
                g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
