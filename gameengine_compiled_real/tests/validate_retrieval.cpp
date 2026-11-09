// =============================================================================
// validate_retrieval - GATE for the retrieval corpus (G13).
//
// The document learner used to be a LOG: every attempt was recorded, and the
// escalation prompt replayed the most RECENT failures. That makes the system
// a one-shot generator with a retry button.
//
// G13 turns it into a CORPUS: the successes become worked examples, and the
// example chosen for a new document is the one whose VOCABULARY matches --
// so the model starts from what the pipeline already learned.
//
// The gate asserts the INVARIANTS of retrieval (not the specific values):
//   - the similarity function is symmetric, bounded, and discriminates
//   - stopwords do not create similarity between unrelated documents
//   - retrieval returns the MOST SIMILAR success, not the most RECENT one
//   - a FAILURE is never offered as a worked example
//   - a document never retrieves its own earlier attempt
//   - a zero-overlap document retrieves nothing
//   - the corpus reaches the NORMAL prompt (not only the escalation prompt)
//   - the corpus survives a save/load round-trip
//   - an older 6-field log still loads (backwards compatibility)
// =============================================================================
#include "delirium/ai/DocLearner.h"

#include <cmath>
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

// Two documents about gearboxes: they share vocabulary.
static const char* kGearboxA =
    "Assembly A-100: a cylindrical housing, 120mm diameter, 80mm tall, with "
    "a rotor of 40mm diameter concentrically mounted inside it. The rotor "
    "sits at the housing's centre and is free to rotate.";
static const char* kGearboxB =
    "Assembly A-200: a gearbox housing with a driven gear inside. The gear "
    "is mounted concentrically within the housing and rotates about the same "
    "shaft.";
static const char* kFemur =
    "The femur is the thigh bone. Its head articulates with the pelvis at the "
    "hip, and its distal end articulates with the tibia at the knee.";
static const char* kUnrelated =
    "The recipe calls for flour, water, salt and yeast. Mix, prove, and bake "
    "until golden brown.";

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

// --- the similarity function --------------------------------------------------

static void test_similarity_properties() {
    // Symmetric: the order of the two texts must not matter.
    CHECK(DocErrorLog::Similarity(kGearboxA, kGearboxB) ==
          DocErrorLog::Similarity(kGearboxB, kGearboxA),
          "similarity must be symmetric");

    // Bounded to [0,1].
    const double self = DocErrorLog::Similarity(kGearboxA, kGearboxA);
    CHECK(self >= 0.0 && self <= 1.0, "self-similarity must be in [0,1]: %f", self);
    CHECK(std::fabs(self - 1.0) < 1e-9,
          "a text must be maximally similar to itself: %f", self);

    // Discriminates: two gearbox documents must outrank a gearbox and a
    // recipe, and a gearbox and a femur.
    const double gear = DocErrorLog::Similarity(kGearboxA, kGearboxB);
    const double bio  = DocErrorLog::Similarity(kGearboxA, kFemur);
    const double food = DocErrorLog::Similarity(kGearboxA, kUnrelated);
    CHECK(gear > bio,  "gearbox-vs-gearbox (%f) must beat gearbox-vs-femur (%f)",
          gear, bio);
    CHECK(gear > food, "gearbox-vs-gearbox (%f) must beat gearbox-vs-recipe (%f)",
          gear, food);
    CHECK(gear > 0.0, "a related pair must score strictly positive: %f", gear);
    CHECK(gear < 1.0,
          "two DIFFERENT documents must not score a perfect 1.0: %f", gear);

    // Two unrelated documents are EQUALLY unrelated, and the measure must
    // not pretend otherwise: a lexical overlap cannot tell "about a femur"
    // from "about a recipe" when neither shares a content word with a
    // gearbox. Asserting a strict order between them would demand a
    // discrimination the measure does not have -- and what actually
    // matters is that BOTH sit below the related pair.
    CHECK(bio <= gear && food <= gear,
          "no unrelated pair may outrank the related pair (%f): bio=%f food=%f",
          gear, bio, food);

    // An empty text has no vocabulary: the score is 0, not a division by
    // zero and not a NaN.
    CHECK(DocErrorLog::Similarity("", kGearboxA) == 0.0,
          "an empty text must score 0");
    CHECK(DocErrorLog::Similarity("the and of", kGearboxA) == 0.0,
          "a stopword-only text must score 0");
}

static void test_stopwords_do_not_create_similarity() {
    // Two documents about completely different things, joined by the same
    // English glue. Without a stopword list these would score high purely
    // on "the"/"of"/"and"/"with", and retrieval would degenerate into
    // "everything is equally similar".
    const std::string a = "the housing and the shaft with the rotor";
    const std::string b = "the femur and the tibia with the patella";
    const double s = DocErrorLog::Similarity(a, b);
    // They do share "and"/"the"/"with" -- all stopwords -- so the only
    // content words are housing/shaft/rotor vs femur/tibia/patella, which
    // share nothing. The score must therefore be 0.
    CHECK(s == 0.0,
          "stopword overlap alone must not create similarity: %f", s);
}

// --- the corpus ---------------------------------------------------------------

// Build a log with: a SUCCESS on gearbox A, a SUCCESS on the femur, and a
// FAILURE on a third gearbox document.
static DocErrorLog BuildCorpus() {
    DocErrorLog log;

    Document a; a.id = "A-100"; a.domain = "machine"; a.text = kGearboxA;
    Document b; b.id = "F-900"; b.domain = "biological"; b.text = kFemur;
    Document c; c.id = "A-300"; c.domain = "machine"; c.text = kGearboxB;

    const auto good = [](const std::string&, const std::string&) {
        return std::string(kTwoLeafTree);
    };
    const auto bad = [](const std::string&, const std::string&) {
        return std::string(kOneLeafTree);
    };

    InterpretDocument(a, good, log);
    InterpretDocument(b, good, log);
    InterpretDocument(c, bad, log);
    return log;
}

static void test_retrieval_picks_the_similar_success() {
    const DocErrorLog log = BuildCorpus();

    // A NEW gearbox document must retrieve the gearbox success (A-100),
    // not the femur one (F-900) and not the failed one (A-300).
    const char* kNewGearbox =
        "A third gearbox, with a rotor inside a gear housing, mounted on a "
        "shaft that rotates inside the housing bore.";
    const auto hits = log.RetrieveSimilar(kNewGearbox, "machine", 3);

    CHECK(!hits.empty(), "a related document must retrieve at least one example");
    if (!hits.empty()) {
        CHECK(hits[0].document_id == "A-100",
              "the most similar success must be retrieved first, got '%s'",
              hits[0].document_id.c_str());
        // The failure must NEVER be offered as a worked example.
        for (const auto& h : hits) {
            CHECK(h.document_id != "A-300",
                  "a FAILED attempt must never be a worked example");
        }
        // The ranked order must be non-increasing in similarity.
        for (size_t i = 1; i < hits.size(); ++i) {
            CHECK(hits[i - 1].similarity >= hits[i].similarity,
                  "the hits must be ordered by descending similarity");
        }
    }
}

static void test_retrieval_is_not_merely_recency() {
    // The corpus must be ordered by SIMILARITY, not by when the attempt
    // happened to be logged. Here the FEMUR is logged LAST (so a recency
    // rule would pick it) but a gearbox query must pick the gearbox.
    DocErrorLog log;
    Document gear; gear.id = "G-1"; gear.domain = "machine"; gear.text = kGearboxA;
    Document bio;  bio.id =  "B-1"; bio.domain = "biological"; bio.text = kFemur;

    const auto good = [](const std::string&, const std::string&) {
        return std::string(kTwoLeafTree);
    };
    InterpretDocument(gear, good, log);   // logged FIRST
    InterpretDocument(bio,  good, log);   // logged LAST

    const auto hits = log.RetrieveSimilar(kGearboxB, "machine", 2);
    CHECK(!hits.empty(), "the corpus must retrieve something");
    if (!hits.empty()) {
        CHECK(hits[0].document_id == "G-1",
              "similarity must beat recency: got '%s', expected 'G-1'",
              hits[0].document_id.c_str());
    }
}

static void test_domain_prior_breaks_a_close_tie() {
    // Two documents with the SAME vocabulary but different domains. The
    // domain bonus must put the same-domain one first, because the SHAPE of
    // the answer differs by domain.
    DocErrorLog log;
    Document m; m.id = "M-1"; m.domain = "machine";     m.text = kGearboxA;
    Document x; x.id = "X-1"; x.domain = "biological";  x.text = kGearboxA;

    const auto good = [](const std::string&, const std::string&) {
        return std::string(kTwoLeafTree);
    };
    InterpretDocument(m, good, log);
    InterpretDocument(x, good, log);

    const auto hits = log.RetrieveSimilar(kGearboxB, "machine", 2);
    CHECK(hits.size() >= 2, "both examples must be candidates: got %zu",
          hits.size());
    if (hits.size() >= 2) {
        CHECK(hits[0].document_id == "M-1",
              "the same-domain example must win a vocabulary tie: got '%s'",
              hits[0].document_id.c_str());
        CHECK(hits[0].same_domain, "the winner must be flagged same-domain");
        CHECK(!hits[1].same_domain, "the loser must not be flagged same-domain");
        // A multiplier, not an addition: the bonus must not exceed 1.25x.
        CHECK(hits[0].similarity <= hits[1].similarity * 1.25 + 1e-9,
              "the domain bonus must be a bounded multiplier: %f vs %f",
              hits[0].similarity, hits[1].similarity);
    }
}

static void test_unrelated_document_retrieves_nothing() {
    const DocErrorLog log = BuildCorpus();
    // A recipe shares no content vocabulary with the corpus. Offering it a
    // worked example would be noise in the prompt.
    const auto hits = log.RetrieveSimilar(kUnrelated, "cooking", 3);
    CHECK(hits.empty(),
          "an unrelated document must retrieve nothing: got %zu", hits.size());
}

static void test_a_document_does_not_retrieve_itself() {
    // A document must not bootstrap from its OWN earlier attempt -- that
    // attempt is already carried by the escalation path, and letting it
    // appear as a "worked example" would let a document teach itself.
    DocErrorLog log;
    Document d; d.id = "S-1"; d.domain = "machine"; d.text = kGearboxA;
    const auto good = [](const std::string&, const std::string&) {
        return std::string(kTwoLeafTree);
    };
    InterpretDocument(d, good, log);

    const auto hits = log.RetrieveSimilar(kGearboxA, "machine", 3);
    for (const auto& h : hits) {
        CHECK(h.document_id != "S-1",
              "a document must not retrieve its own prior attempt");
    }
}

static void test_failures_are_never_worked_examples() {
    const DocErrorLog log = BuildCorpus();
    // The failure's document (A-300, a gearbox) IS similar to a gearbox
    // query -- but it failed, so it must not be offered.
    const auto hits = log.RetrieveSimilar(kGearboxB, "machine", 5);
    for (const auto& h : hits) {
        CHECK(h.document_id != "A-300",
              "a failed attempt must never be retrieved");
    }
    // And the retrieval must not be empty just because the failure was
    // excluded -- the OTHER gearbox success is still there.
    CHECK(!hits.empty(),
          "excluding a failure must not empty the corpus");
}

// --- the corpus reaches the prompt --------------------------------------------

static void test_corpus_reaches_the_normal_prompt() {
    // The decisive invariant: the corpus must reach the FIRST attempt, not
    // only the escalation. If it only reaches the escalation, escalation is
    // doing all the work and the system is a one-shot generator.
    DocErrorLog log;
    Document first; first.id = "A-100"; first.domain = "machine"; first.text = kGearboxA;
    const auto good = [](const std::string&, const std::string&) {
        return std::string(kTwoLeafTree);
    };
    InterpretDocument(first, good, log);

    // A second, similar document. The NORMAL attempt must now SEE the
    // worked example.
    std::string seen_user;
    const auto ask = [&](const std::string&, const std::string& user) {
        seen_user = user;
        return std::string(kTwoLeafTree);
    };
    Document second; second.id = "A-200"; second.domain = "machine"; second.text = kGearboxB;
    DocResult r = InterpretDocument(second, ask, log, /*escalate=*/false);

    CHECK(r.ok, "the second attempt must succeed: %s", r.reason.c_str());
    CHECK(seen_user.find("WORKED EXAMPLES") != std::string::npos,
          "the NORMAL prompt must carry the retrieved worked examples");
    CHECK(seen_user.find("A-100") != std::string::npos,
          "the normal prompt must name the retrieved document");
    CHECK(seen_user.find(kTwoLeafTree) != std::string::npos,
          "the normal prompt must carry the retrieved TREE, not just its id");
}

static void test_empty_corpus_adds_nothing_to_the_prompt() {
    // A corpus with NO successes has nothing to retrieve, and the prompt
    // must not gain a stray empty section.
    //
    // NOTE: the check must run BEFORE any successful attempt is recorded.
    // An earlier version of this test called InterpretDocument first (which
    // SUCCEEDS and populates the corpus) and then asserted the context was
    // empty -- which is only true if the retrieval fails, not if the corpus
    // is empty. The two were conflated.
    DocErrorLog log;
    const std::string ctx = log.AsRetrievedContext(kGearboxB, "machine", 3);
    CHECK(ctx.empty(),
          "an empty corpus must produce empty context, got %zu chars", ctx.size());

    // And a corpus whose only entry is a FAILURE also retrieves nothing:
    // a worked example must be a tree that passed.
    const auto bad = [](const std::string&, const std::string&) {
        return std::string(kOneLeafTree);
    };
    Document d; d.id = "A-300"; d.domain = "machine"; d.text = kGearboxB;
    InterpretDocument(d, bad, log);
    const std::string ctx2 = log.AsRetrievedContext(kGearboxB, "machine", 3);
    CHECK(ctx2.empty(),
          "a corpus with only failures must produce empty context, got %zu chars",
          ctx2.size());
}

// --- persistence --------------------------------------------------------------

static void test_corpus_survives_a_roundtrip() {
    DocErrorLog log = BuildCorpus();

    const std::string path = "/tmp/hermes-retrieval-corpus.txt";
    CHECK(log.Save(path), "the corpus must save");

    DocErrorLog reloaded;
    CHECK(reloaded.Load(path), "the corpus must load");
    CHECK(reloaded.Size() == log.Size(), "the entry count must survive");
    CHECK(reloaded.SuccessCount() == log.SuccessCount(),
          "the successes must survive");
    CHECK(reloaded.FailureCount() == log.FailureCount(),
          "the failures must survive");

    // The DOCUMENT TEXT must survive -- that is what makes the reloaded
    // log a corpus rather than a log.
    const auto before = log.RetrieveSimilar(kGearboxB, "machine", 3);
    const auto after = reloaded.RetrieveSimilar(kGearboxB, "machine", 3);
    CHECK(before.size() == after.size(),
          "the retrieval must return the same count after a round-trip");
    if (!before.empty() && !after.empty()) {
        CHECK(before[0].document_id == after[0].document_id,
              "the retrieval must return the same example after a round-trip");
        CHECK(std::fabs(before[0].similarity - after[0].similarity) < 1e-12,
              "the similarity must survive the round-trip");
    }
}

static void test_an_older_six_field_log_still_loads() {
    // BACKWARDS COMPATIBILITY: a log written before G13 has six fields.
    // It must still load -- otherwise the corpus is unreadable until it
    // is rebuilt, and every accumulated failure is lost.
    const std::string path = "/tmp/hermes-retrieval-legacy.txt";
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f << "FAIL\t---\tLEG-1\tvalidate\tsome reason\tnot json at all\n";
        f << "OK\t---\tLEG-2\t\t\tkTwoLeafTree\n";
    }
    DocErrorLog log;
    CHECK(log.Load(path), "a legacy six-field log must load");
    CHECK(log.Size() == 2, "both legacy entries must load: got %zu", log.Size());
    CHECK(log.FailureCount() == 1, "the legacy failure must load");
    CHECK(log.SuccessCount() == 1, "the legacy success must load");
    // A legacy entry has no document text, so it has no vocabulary -- it
    // must not be retrieved, and must not crash the ranker.
    const auto hits = log.RetrieveSimilar(kGearboxB, "machine", 3);
    CHECK(hits.empty(),
          "a legacy entry without a document must not be retrieved: got %zu",
          hits.size());
}

int main() {
    test_similarity_properties();
    test_stopwords_do_not_create_similarity();
    test_retrieval_picks_the_similar_success();
    test_retrieval_is_not_merely_recency();
    test_domain_prior_breaks_a_close_tie();
    test_unrelated_document_retrieves_nothing();
    test_a_document_does_not_retrieve_itself();
    test_failures_are_never_worked_examples();
    test_corpus_reaches_the_normal_prompt();
    test_empty_corpus_adds_nothing_to_the_prompt();
    test_corpus_survives_a_roundtrip();
    test_an_older_six_field_log_still_loads();

    std::printf("\nvalidate_retrieval: %d checks, %d failures\n",
                g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
