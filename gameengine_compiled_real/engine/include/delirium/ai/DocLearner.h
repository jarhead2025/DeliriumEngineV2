// FILEPATH: engine/include/delirium/ai/DocLearner.h
// Document-driven reconstruction: technical documents -> element tree.
//
// THE PROBLEM THIS SOLVES:
//   A closed body (a sphere, a motor housing, a femur) cannot be expressed
//   by ONE graph substance -- the substance interface maps local (x,y) to a
//   surface point, so it can only ever describe the TOP sheet. A sphere
//   needs several sheets arranged in 3D, and WHICH sheets, and their
//   proportions, is knowledge that has to come from somewhere.
//
// WHY DOCUMENTS:
//   A technical document (a machine's exploded-view parts list, a patent
//   description, an anatomical atlas entry) already encodes that knowledge
//   in a structured way: named parts, their relationships, their relative
//   proportions, how they connect. The local model reads the document and
//   emits the element tree.
//
// THE LEARNING LOOP:
//   1. READ the document (text) -- the "training example".
//   2. ASK the local model to emit an element tree from it.
//   3. VALIDATE the tree (the existing parser + baker's out-of-support
//      guard + a geometry check).
//   4. LOG every failure with its reason. The log is the dataset.
//   5. ESCALATE: when the local model cannot produce a valid tree, re-ask
//      with the failure reason AND the accumulated error log attached, so
//      the model can see the pattern of what went wrong. A larger model
//      can be consulted with the same enriched context.
//
// WHY THE ERROR LOG IS THE POINT:
//   The error log is what turns a one-shot prompt into a LEARNING system:
//   every rejected tree is a labelled example of a failure, and every
//   accepted tree is a labelled example of success. Over time the log is
//   the corpus from which the categorizer is trained. Without it, each
//   failed generation is simply lost.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "delirium/ai/ElementTreeParser.h"

namespace DeliRium {
namespace AI {

// One attempt at turning a document into a tree.
struct DocAttempt {
    // The document that was interpreted.
    std::string document_id;
    // The document's TEXT.
    //
    // WHY THIS IS IN THE LOG (G13): the corpus is a RETRIEVAL corpus. To
    // find the prior document most similar to the one being interpreted, the
    // log has to be able to compare documents -- and a document id is a
    // label, not a content. Without the text, "nearest prior document" is
    // not computable and the success log can only ever be replayed by
    // RECENCY, which is the weakest possible selection rule.
    std::string document_text;
    // The domain the document belongs to ("machine", "biological", ...).
    // A same-domain document is a better worked example than a
    // cross-domain one even when the words overlap less.
    std::string domain;
    // The model's raw reply.
    std::string raw_reply;
    // The GEOMETRIC TOOL TRANSCRIPT -- what the model asked for, and what
    // each tool answered, in order.
    //
    // It is the evidence the learner needs. A successful document becomes a
    // worked example carrying the tools it used (so the next attempt can
    // imitate the SEQUENCE, not just the answer); a failure carries the
    // tools it TRIED (so the escalation prompt can say "you called
    // resolve_torus_radii with outer_radius, which that tool does not
    // accept").
    //
    // Empty when the tools were not used.
    std::string tool_transcript;
    // The stage that failed (empty on success).
    //   "parse"        -- the reply was not a valid element tree
    //   "validate"     -- the tree was structurally invalid
    //   "bake"         -- the tree could not be baked into a body
    //   "geometry"     -- the baked body did not match the document
    //   "transport"    -- the model could not be reached at all
    std::string failed_stage;
    // The human-readable reason for the failure.
    std::string reason;
    // True if the attempt produced a usable tree.
    bool ok = false;
};

// The accumulating log of every attempt.
//
// It is APPEND-ONLY and survives across sessions (it is written to disk),
// because its value is cumulative: the error patterns are what the
// escalation prompt and the future categorizer are built from.
class DocErrorLog {
public:
    // Record an attempt. `escalated` marks an attempt that was retried
    // with a larger model / enriched context.
    void Record(const DocAttempt& a, bool escalated = false);

    // The errors, oldest first, as compact prompt fragments. Each is
    // "document=<id> stage=<stage>: <reason>" -- short enough to fit in
    // a prompt, specific enough to be actionable.
    std::string AsPromptContext(size_t max_entries = 12) const;

    // The successful attempts, as compact prompt fragments -- these are
    // the POSITIVE examples, which matter as much as the negative ones.
    std::string AsSuccessContext(size_t max_entries = 6) const;

    // -----------------------------------------------------------------------
    // RETRIEVAL (G13): the corpus.
    //
    // "Give me the past success most like THIS document" is the difference
    // between a log and a corpus. Recency is the weakest possible rule: it
    // picks the example that happened to be logged last, which has no
    // relationship to whether it teaches anything. Retrieval picks the
    // example that SHARES VOCABULARY with the document being read, so the
    // model sees a worked example of the shape it is about to produce.
    //
    // The similarity is a token-overlap score (Jaccard-style) over the
    // document's words. It is deliberately simple and dependency-free:
    //   - no embeddings, no model, no network -- the corpus works offline;
    //   - deterministic, so a retrieval is reproducible and testable;
    //   - explainable ("these words overlap"), which matters when the
    //     corpus is a prompt that a human is meant to be able to audit.
    //
    // A DOMAIN BONUS applies: a prior document in the same domain outranks
    // a cross-domain one even when the raw overlap is lower, because the
    // SHAPE of the answer differs by domain (a machine wants parts and
    // joints; a biological structure wants bones and articulations).
    // -----------------------------------------------------------------------

    // One retrieved worked example.
    struct Retrieved {
        std::string document_id;
        std::string document_text;   // the prior document (truncated)
        std::string raw_reply;       // the WORKED EXAMPLE: a validated tree
        double       similarity = 0.0;  // 0..1, the token-overlap score
        bool         same_domain = false;
    };

    // The successful attempts most similar to `document_text`, best first.
    // `max_entries` bounds how many are returned. `domain` (may be empty)
    // biases the ranking toward the same domain. Only SUCCESSFUL attempts
    // are candidates -- a worked example must be a tree that passed.
    std::vector<Retrieved> RetrieveSimilar(const std::string& document_text,
                                            const std::string& domain,
                                            size_t max_entries = 3) const;

    // The retrieved worked examples, formatted as prompt context.
    // This is what the NORMAL (non-escalated) attempt receives, so the
    // model starts from prior knowledge instead of from nothing.
    std::string AsRetrievedContext(const std::string& document_text,
                                   const std::string& domain,
                                   size_t max_entries = 3) const;

    // The token-overlap similarity between two texts, in [0,1].
    // Exposed so the ranking is testable on its own -- a similarity
    // function that is only observable through a whole pipeline is a
    // similarity function nobody can debug.
    static double Similarity(const std::string& a, const std::string& b);

    size_t Size() const { return entries_.size(); }
    size_t FailureCount() const;
    size_t SuccessCount() const;

    // The attempts, oldest first (for inspection / the UI). Only the
    // attempt records -- the escalation flag is metadata about the entry.
    std::vector<DocAttempt> Attempts() const {
        std::vector<DocAttempt> out;
        out.reserve(entries_.size());
        for (const Entry& e : entries_) out.push_back(e.attempt);
        return out;
    }
    // How many of the logged attempts were escalated.
    size_t EscalatedCount() const;

    // Persist to / load from a text file. The format is one record per
    // line, tab-separated, with newlines in the fields escaped -- so the
    // log stays greppable and diffable.
    //
    // The record carries the document TEXT and DOMAIN as well as the reply
    // (G13): the corpus must survive a restart, and a corpus that forgets
    // its documents on restart is just a log again.
    bool Save(const std::string& path) const;
    bool Load(const std::string& path);

private:
    struct Entry {
        DocAttempt attempt;
        bool escalated = false;
    };
    std::vector<Entry> entries_;
};

// The prompt that asks a model to turn a document into an element tree.
//
// The document is asked for THREE things, because the element schema needs
// all three:
//   - the PARTS (each becomes a system, with its own substance)
//   - the RELATIONS (which part contains/attaches to which -> the
//     element tree's parent/child structure and the origins)
//   - the PROPORTIONS (the scale of each part relative to the whole)
//
// The prompt also states the CLOSED-BODY RULE explicitly, because the
// model otherwise emits a single sheet and the result is "only one face".
std::string DocToTreeSystemPrompt();

// The prompt for the ESCALATED attempt: the original document, the
// failure reason, and the accumulated error log, so the model can see what
// has already failed and why.
std::string DocToTreeEscalationPrompt(const std::string& document,
                                      const std::string& prior_failure,
                                      const std::string& error_context,
                                      const std::string& success_context);

// A document to interpret. `text` is the document's content; `id`
// identifies it in the log.
struct Document {
    std::string id;
    std::string text;
    // The DOMAIN the document belongs to. The user asked to start with
    // technical documents from machines and move up to biological, so the
    // domain is explicit and the log can be filtered by it.
    std::string domain = "machine";
};

// The result of interpreting a document.
struct DocResult {
    bool ok = false;
    std::unique_ptr<Element> tree;
    std::string reason;       // why it failed (empty on success)
    std::string failed_stage; // which stage failed
    bool escalated = false;   // true if this was the escalation attempt
};

// Interpret a document with a model.
//
// `ask` is a callback the caller supplies: it takes (system_prompt,
// user_message) and returns the model's reply (or an empty string on a
// transport failure). The learner does not own the transport -- the
// workspace owns the LLMClient, and the probe owns its own. That keeps
// this component testable without a server.
//
// The escalation ladder:
//   attempt 1: the local model, with the plain prompt.
//   attempt 2 (escalate=true): the local model again, but with the
//              failure reason and the error log attached.
//
// `larger_model_ask` is an optional second callback for a LARGER model
// (a remote endpoint, a stronger local model). When the local model has
// already failed and `larger_model_ask` is provided, the ESCALATED
// attempt goes to the larger model instead -- that is the "consult a
// larger model if necessary" path.
//
// -----------------------------------------------------------------------------
// THE GEOMETRIC TOOLS (opt-in)
// -----------------------------------------------------------------------------
//
// `use_geo_tools` makes the learner run the model through the TOOL LOOP
// (LLM::RunToolLoop) with the engine's geometric analysis tools, so the
// model can compute the arithmetic it gets wrong -- the torus's radii, a
// part's scale, the closed-body rule -- instead of guessing it.
//
// It is OPT-IN and the reason is measurement: the tools change what the
// model produces, so a caller must be able to run both ways and compare.
// The default (off) preserves the previous behaviour exactly.
//
// When the tools are used, the tool transcript is recorded in the attempt
// (`tool_transcript`). That is the evidence a learner needs: a successful
// document becomes a worked example carrying the tools it used, and a
// failure carries the tools it TRIED.
DocResult InterpretDocument(
    const Document& doc,
    const std::function<std::string(const std::string& system,
                                    const std::string& user)>& ask,
    DocErrorLog& log,
    bool escalate = false,
    const std::function<std::string(const std::string& system,
                                    const std::string& user)>* larger_model_ask = nullptr,
    bool use_geo_tools = false);

} // namespace AI
} // namespace DeliRium
