// FILEPATH: engine/src/ai/DocLearner.cpp
// Document-driven reconstruction: technical documents -> element tree.
//
// See engine/include/delirium/ai/DocLearner.h for the design and the
// reasons behind it.
#include "delirium/ai/DocLearner.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace DeliRium {
namespace AI {

namespace {

// Escape a field so the log stays one-record-per-line.
std::string EscapeField(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (const char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;       break;
        }
    }
    return out;
}

std::string UnescapeField(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '\\' || i + 1 >= s.size()) { out += s[i]; continue; }
        switch (s[++i]) {
            case 'n':  out += '\n'; break;
            case 'r':  out += '\r'; break;
            case 't':  out += '\t'; break;
            case '\\': out += '\\'; break;
            default:   out += s[i]; break;
        }
    }
    return out;
}

// Split on a single tab.
std::vector<std::string> SplitTabs(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    for (const char c : line) {
        if (c == '\t') { out.push_back(cur); cur.clear(); }
        else cur += c;
    }
    out.push_back(cur);
    return out;
}

// -----------------------------------------------------------------------------
// Tokenisation for the retrieval similarity.
//
// WHY WORDS AND NOT SUBSTRINGS: a substring match scores "rot" against
// "rotor" and "rotation" and "rotational" identically, which is exactly
// the distinction the corpus needs to make ("a rotor" is about the same
// kind of thing as "another rotor"; it is not about "rotation" the act).
//
// WHY A STOPWORD LIST: the words that carry the most overlap in technical
// documents are the ones that carry the least information -- "the", "a",
// "of", "is", "with", "and", "in", "to", "that", "it", "at", "by", "for",
// "on", "as", "from". Left in, two documents about completely different
// machines ("the housing and the shaft") score a high overlap on "the",
// "and" and "of" alone, and the retrieval degenerates into "all documents
// are equally similar" -- which is the same as recency with extra steps.
//
// The list is small and domain-neutral on purpose: a domain-specific
// stopword list would need to be maintained per corpus, and the cost of a
// slightly worse score is far lower than the cost of a corpus that
// silently stops discriminating.
const char* const kStopwords[] = {
    "the", "a", "an", "and", "or", "of", "is", "are", "was", "were", "be",
    "been", "being", "to", "in", "on", "at", "by", "for", "with", "from",
    "as", "that", "this", "these", "those", "it", "its", "which", "has",
    "have", "had", "not", "no", "but", "into", "than", "then", "so", "such",
    "each", "other", "any", "all", "can", "will", "would", "there", "their",
    "they", "we", "you", "our", "its", "mm", "cm", "kg", "ms", "rpm",
};

// Lowercase, split on non-alphanumerics, drop stopwords and 1-char tokens.
std::vector<std::string> Tokenize(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    auto flush = [&]() {
        if (cur.empty()) return;
        // Lowercase (ASCII only -- the corpus is technical English and a
        // Unicode-aware case fold would drag in a dependency for nothing).
        for (char& c : cur) {
            if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
        }
        if (cur.size() > 1) {
            bool stop = false;
            for (const char* sw : kStopwords) {
                if (cur == sw) { stop = true; break; }
            }
            if (!stop) out.push_back(cur);
        }
        cur.clear();
    };
    for (const char c : text) {
        const bool alnum = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                           (c >= '0' && c <= '9');
        if (alnum) cur += c;
        else flush();
    }
    flush();
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// DocErrorLog
// ---------------------------------------------------------------------------

void DocErrorLog::Record(const DocAttempt& a, bool escalated) {
    entries_.push_back(Entry{a, escalated});
}

size_t DocErrorLog::FailureCount() const {
    size_t n = 0;
    for (const Entry& e : entries_) if (!e.attempt.ok) ++n;
    return n;
}

size_t DocErrorLog::SuccessCount() const {
    size_t n = 0;
    for (const Entry& e : entries_) if (e.attempt.ok) ++n;
    return n;
}

size_t DocErrorLog::EscalatedCount() const {
    size_t n = 0;
    for (const Entry& e : entries_) if (e.escalated) ++n;
    return n;
}

std::string DocErrorLog::AsPromptContext(size_t max_entries) const {
    // The MOST RECENT failures are the most relevant -- a model that just
    // saw the last failure is the one that can avoid repeating it.
    std::string out;
    size_t shown = 0;
    for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
        if (it->attempt.ok) continue;
        if (shown >= max_entries) break;
        out += "FAILED document=" + it->attempt.document_id;
        out += " stage=" + it->attempt.failed_stage;
        out += ": " + it->attempt.reason + "\n";
        ++shown;
    }
    return out;
}

std::string DocErrorLog::AsSuccessContext(size_t max_entries) const {
    // The most recent SUCCESSES, as positive examples. These matter as
    // much as the failures: they show the model the shape of a valid
    // answer for THIS pipeline.
    std::string out;
    size_t shown = 0;
    for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
        if (!it->attempt.ok) continue;
        if (shown >= max_entries) break;
        out += "OK document=" + it->attempt.document_id;
        // The successful reply, truncated: it is the WORKED EXAMPLE, and
        // a worked example is worth more than a description of one.
        std::string raw = it->attempt.raw_reply;
        if (raw.size() > 600) raw = raw.substr(0, 600) + "...";
        out += ":\n" + raw + "\n";
        ++shown;
    }
    return out;
}

bool DocErrorLog::Save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    for (const Entry& e : entries_) {
        f << (e.attempt.ok ? "OK" : "FAIL") << '\t'
          << (e.escalated ? "ESC" : "---") << '\t'
          << EscapeField(e.attempt.document_id) << '\t'
          << EscapeField(e.attempt.failed_stage) << '\t'
          << EscapeField(e.attempt.reason) << '\t'
          << EscapeField(e.attempt.raw_reply) << '\t'
          << EscapeField(e.attempt.document_text) << '\t'
          << EscapeField(e.attempt.domain) << '\n';
    }
    return f.good();
}

bool DocErrorLog::Load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    entries_.clear();
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        const std::vector<std::string> f8 = SplitTabs(line);
        if (f8.size() < 6) continue;
        DocAttempt a;
        a.ok = (f8[0] == "OK");
        a.document_id = UnescapeField(f8[2]);
        a.failed_stage = UnescapeField(f8[3]);
        a.reason = UnescapeField(f8[4]);
        a.raw_reply = UnescapeField(f8[5]);
        // The document text and domain are OPTIONAL trailing fields: a log
        // written by an older build has only six, and it must still load
        // (otherwise the corpus would be unreadable until it was rebuilt).
        if (f8.size() >= 8) {
            a.document_text = UnescapeField(f8[6]);
            a.domain = UnescapeField(f8[7]);
        }
        entries_.push_back(Entry{a, f8[1] == "ESC"});
    }
    return true;
}

// -----------------------------------------------------------------------------
// RETRIEVAL
// -----------------------------------------------------------------------------

double DocErrorLog::Similarity(const std::string& a, const std::string& b) {
    const std::vector<std::string> ta = Tokenize(a);
    const std::vector<std::string> tb = Tokenize(b);
    if (ta.empty() || tb.empty()) return 0.0;

    // Sets, not multisets: a document that says "rotor" eight times is not
    // eight times more about rotors than one that says it once.
    std::vector<std::string> ua = ta, ub = tb;
    std::sort(ua.begin(), ua.end());
    ua.erase(std::unique(ua.begin(), ua.end()), ua.end());
    std::sort(ub.begin(), ub.end());
    ub.erase(std::unique(ub.begin(), ub.end()), ub.end());

    // Intersection size (both sorted -> a linear merge).
    size_t i = 0, j = 0, common = 0;
    while (i < ua.size() && j < ub.size()) {
        if (ua[i] == ub[j]) { ++common; ++i; ++j; }
        else if (ua[i] < ub[j]) ++i;
        else ++j;
    }
    // Jaccard: |A n B| / |A u B|. Chosen over "intersection over min" so a
    // short query that matches a long document is NOT scored at 1.0 --
    // otherwise a two-word query would match every document that happens to
    // contain those two words and the ranking would be meaningless.
    const double denom = double(ua.size() + ub.size() - common);
    if (denom <= 0.0) return 0.0;
    return double(common) / denom;
}

std::vector<DocErrorLog::Retrieved> DocErrorLog::RetrieveSimilar(
    const std::string& document_text, const std::string& domain,
    size_t max_entries) const {
    std::vector<Retrieved> hits;

    for (const Entry& e : entries_) {
        // Only SUCCESSES are candidates. A worked example has to be a tree
        // that passed validation -- teaching the model from a tree the
        // pipeline REJECTED would teach it the failure mode.
        if (!e.attempt.ok) continue;
        // A prior attempt on the SAME document is not a worked example for
        // this one -- it is this document's own earlier attempt, and it is
        // already carried by the escalation path. Including it would let a
        // document bootstrap itself from its own output.
        if (!e.attempt.document_text.empty() &&
            e.attempt.document_text == document_text) {
            continue;
        }

        Retrieved r;
        r.document_id = e.attempt.document_id;
        r.document_text = e.attempt.document_text;
        r.raw_reply = e.attempt.raw_reply;
        r.similarity = Similarity(document_text, e.attempt.document_text);
        r.same_domain = !domain.empty() && !e.attempt.domain.empty() &&
                        domain == e.attempt.domain;
        hits.push_back(std::move(r));
    }

    // Rank. The domain bonus is a MULTIPLIER on the similarity, not an
    // addition: an addition could let a zero-overlap cross-domain example
    // outrank a strongly-overlapping one, which is the opposite of what
    // the domain prior is for.
    //
    // The bonus is 1.25 -- enough to break a close tie, not enough to
    // promote an unrelated document over a clearly-related one.
    for (Retrieved& r : hits) {
        if (r.same_domain) r.similarity *= 1.25;
    }
    std::stable_sort(hits.begin(), hits.end(),
                     [](const Retrieved& a, const Retrieved& b) {
                         return a.similarity > b.similarity;
                     });

    // A similarity of zero means "shares no vocabulary at all" -- that is
    // not a worked example, it is noise in the prompt. Drop those.
    hits.erase(std::remove_if(hits.begin(), hits.end(),
                              [](const Retrieved& r) { return r.similarity <= 0.0; }),
               hits.end());

    if (hits.size() > max_entries) hits.resize(max_entries);
    return hits;
}

std::string DocErrorLog::AsRetrievedContext(const std::string& document_text,
                                             const std::string& domain,
                                             size_t max_entries) const {
    const std::vector<Retrieved> hits =
        RetrieveSimilar(document_text, domain, max_entries);
    if (hits.empty()) return std::string();

    std::string out;
    out += "WORKED EXAMPLES FROM SIMILAR DOCUMENTS (these were accepted by "
           "the validator -- reuse their shape):\n";
    for (const Retrieved& r : hits) {
        out += "\n--- document " + r.document_id;
        if (r.same_domain) out += " (same domain)";
        out += " ---\n";
        // The prior document, truncated: it is context, not the payload.
        std::string prior = r.document_text;
        if (prior.size() > 400) prior = prior.substr(0, 400) + "...";
        if (!prior.empty()) {
            out += "ITS DOCUMENT:\n" + prior + "\n";
        }
        // The WORKED EXAMPLE: the validated tree. This is the payload.
        std::string tree = r.raw_reply;
        if (tree.size() > 1200) tree = tree.substr(0, 1200) + "...";
        out += "ITS ACCEPTED ANSWER:\n" + tree + "\n";
    }
    return out;
}

// ---------------------------------------------------------------------------
// The prompts
// ---------------------------------------------------------------------------

std::string DocToTreeSystemPrompt() {
    // The element schema itself (single source of truth -- the same text
    // the workspace uses), plus the document-specific instructions.
    return ElementTreeSystemPrompt() + R"(

=== READING A TECHNICAL DOCUMENT ===

You are given a DOCUMENT that describes a real object: a machine, an
assembly, an anatomical structure, or any other physical thing. Produce an
element tree that reconstructs that object.

WORK IN THREE PASSES, and emit only the final JSON:

  1. PARTS. List the document's named parts. Each becomes one system in
     the tree. A part that is a simple rounded mass uses "ellipsoid"; a
     part that is a ring or a frame uses "torus"; a part that is a flat
     plate or a face uses "plane"; a part that has a shaped surface (a
     gear tooth, a fin, a blade, a cam profile) uses "heightfield".

  2. RELATIONS. Each part's position and size come from the document's
     description of how the parts fit together: "the rotor sits inside the
     housing" -> the rotor is a CHILD of the housing, with its origin at
     the housing's centre and its scale smaller. Use the tree's
     parent/child structure for containment and attachment, and use each
     element's "origin" and "scale" for placement.

  3. PROPORTIONS. Sizes are RELATIVE. Make the whole object fit roughly
     within [-0.5, 0.5] on each axis, and size each part relative to the
     others according to the document. If the document gives real
     dimensions, use their RATIOS, not their absolute values.

=== READ IT AS A MECHANISM, NOT A PILE OF SHAPES ===

A technical document describes an ASSEMBLY: parts, the joints between them,
what each part is made of, and how they move relative to each other. A
tree that is only a pile of overlapping ellipsoids loses all of that -- and
the parts are the only thing that makes the reconstruction a MECHANISM.

So, for a MACHINE or an ANATOMICAL structure:

  - Give each MOVING part a BONE. A bone is a rigid segment: its "head"
    is the pivot, its "tail" is the far end, and its "parent" is the part
    it pivots against (by NAME). The rotor's bone has the housing's bone
    as its parent. A limb's bone has the torso's bone as its parent.
  - NAME THE JOINTS. If the document says the knee, the hip, the bearing,
    the pin, give the bone a "joints" entry with that name, so the
    articulation can be referred to by NAME and not by index.
  - GIVE EACH PART ITS MATERIAL. A document that says "steel housing" or
    "ABS cover" is stating a physical fact. Put the material on the part's
    bone: it is what makes the reconstruction carry real permeability,
    conductivity and heat capacity rather than being a shape.

A part that has a substance (its shape) AND a bone (how it moves) AND a
material (what it is made of) is a COMPLETE part. Prefer that.

Do NOT invent a material the document did not name, and do NOT add a bone
to a part the document does not describe as moving. An honest omission is
better than a plausible invention.

=== THE CLOSED-BODY RULE (READ THIS) ===

Each substance describes only the UPPER SHEET of a surface -- it is a graph
z = f(x, y). So ONE substance is NEVER a closed body. A sphere is not one
ellipsoid: it is an ellipsoid PLUS another one placed below it. A housing
is not one ellipsoid: it is several parts.

To close a body, compose sheets. The rule is MEASURED (see
examples/ai_pipeline_probe/closed_body_probe.cpp), not guessed:

  world.z = origin.z + local.z * scale.z
  with local.z spanning [0, sz] for an ellipsoid:
     scale.z = +1  ->  the sheet spans [origin.z, origin.z + sz]   (UP)
     scale.z = -1  ->  the sheet spans [origin.z - sz, origin.z]   (DOWN)

A negative z scale ALREADY flips the sheet downward from the seam plane it
shares with its parent. So the lower sheet of a closed sphere keeps the SAME
origin as the upper one and only flips the scale:

  a sphere = an ellipsoid + an ellipsoid with scale z NEGATIVE and the
             SAME origin ([0,0,0]). They meet at z = 0 and the body
             spans exactly [-sz, +sz].

Do NOT offset the lower sheet's origin: that shifts it off the seam and
leaves a GAP (the body does not close). Do NOT leave the scale positive:
that closes the geometry but turns the lower sheet INSIDE-OUT (its normals
still point up, so it renders as a hollow shell).

  CORRECT   upper: {"kind":"body","substance":"ellipsoid",
                    "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
                    "children":[{"kind":"lower","substance":"ellipsoid",
                                 "substance_params":{"sx":0.5,"sy":0.5,"sz":0.5},
                                 "scale":[1,1,-1],"origin":[0,0,0]}]}

  WRONG (gap)      the lower sheet's origin is offset to [0,0,-0.5]
  WRONG (inside)   the lower sheet's scale is [1,1,1] (not flipped)

Other closures:
  - a cylinder = an ellipsoid (elongated in z) + a plane at each end
  - a ring/housing = a torus, optionally with ellipsoids inside it
Each element's "origin" places it in the PARENT's local frame, and
"scale" scales it. A child with a negative z scale flips the sheet.

=== HOW TO FLIP A SHEET (THE ONE MISTAKE THAT MATTERS MOST) ===

To point a sheet DOWNWARD, use the element's "scale" field with a negative
component. NEVER use a negative half-extent.

  CORRECT   {"kind":"lower", "substance":"ellipsoid",
             "substance_params":{"sx":0.6,"sy":0.6,"sz":0.5},
             "scale":[1,1,-1], "origin":[0,0,0]}
  WRONG     {"kind":"lower", "substance":"ellipsoid",
             "substance_params":{"sx":0.6,"sy":0.6,"sz":-1}}

"substance_params" holds SIZES and they are always positive. The
"scale" field holds the ORIENTATION and the placement. A negative
"sx"/"sy"/"sz" is rejected.

=== OUTPUT ===
Emit ONLY the JSON object. No prose, no explanation, no markdown fence.
)";
}

std::string DocToTreeEscalationPrompt(const std::string& document,
                                      const std::string& prior_failure,
                                      const std::string& error_context,
                                      const std::string& success_context) {
    std::string out;
    out += "A previous attempt to interpret this document FAILED.\n\n";
    out += "THE DOCUMENT:\n";
    out += document;
    out += "\n\n";
    if (!prior_failure.empty()) {
        out += "THE PREVIOUS FAILURE:\n";
        out += prior_failure;
        out += "\n\n";
    }
    if (!error_context.empty()) {
        out += "PREVIOUS FAILURES ON OTHER DOCUMENTS (do not repeat them):\n";
        out += error_context;
        out += "\n";
    }
    if (!success_context.empty()) {
        out += "WORKED EXAMPLES OF A VALID ANSWER:\n";
        out += success_context;
        out += "\n";
    }
    out += "Retry, following the schema and the closed-body rule exactly.\n";
    out += "Emit ONLY the JSON object. No prose, no markdown fence.\n";
    return out;
}

// ---------------------------------------------------------------------------
// InterpretDocument
// ---------------------------------------------------------------------------

DocResult InterpretDocument(
    const Document& doc,
    const std::function<std::string(const std::string& system,
                                    const std::string& user)>& ask,
    DocErrorLog& log,
    bool escalate,
    const std::function<std::string(const std::string& system,
                                    const std::string& user)>* larger_model_ask)
{
    DocResult res;

    // Which model answers this attempt. The escalation ladder:
    //   - the local model by default;
    //   - a LARGER model when escalating and one is available.
    const bool use_larger = escalate && larger_model_ask && *larger_model_ask;
    const auto& asker = use_larger ? *larger_model_ask : ask;

    // The SYSTEM prompt. On escalation the schema is unchanged (it is the
    // contract) but the USER message carries the failure context.
    const std::string system = DocToTreeSystemPrompt();

    std::string user;
    if (!escalate) {
        // The NORMAL attempt also gets the corpus. This is the whole point
        // of G13: the first attempt should start from what the pipeline has
        // already learned, not from nothing. The escalation path exists for
        // when that is not enough -- but if the normal path never gets the
        // corpus, escalation is doing all the work and the system is a
        // one-shot generator with a retry button.
        const std::string retrieved = log.AsRetrievedContext(doc.text, doc.domain);
        user = "DOCUMENT (" + doc.domain + ", id=" + doc.id + "):\n";
        user += doc.text;
        if (!retrieved.empty()) {
            user += "\n\n";
            user += retrieved;
        }
        user += "\n\nReconstruct this object as an element tree.\n";
    } else {
        std::string prior;
        // The most recent failure for THIS document is the one to fix.
        for (auto it = log.Attempts().rbegin(); it != log.Attempts().rend(); ++it) {
            if (!it->ok && it->document_id == doc.id) {
                prior = "stage=" + it->failed_stage + ": " + it->reason;
                break;
            }
        }
        // The worked examples for the escalation are the RETRIEVED ones (the
        // successes whose vocabulary matches this document), not the most
        // recent ones. The normal attempt uses retrieval; the escalation
        // using recency would mean the retry is taught by a DIFFERENT and
        // weaker rule than the first attempt -- and a retry that is taught
        // worse than the original is not a retry, it is a fresh attempt
        // with extra text.
        //
        // The recency fallback matters: on a cold corpus (nothing retrieved)
        // the recent successes are better than nothing.
        std::string successes = log.AsRetrievedContext(doc.text, doc.domain);
        if (successes.empty()) successes = log.AsSuccessContext();
        user = DocToTreeEscalationPrompt(doc.text, prior,
                                         log.AsPromptContext(),
                                         successes);
    }

    // --- TRANSPORT ---------------------------------------------------------
    const std::string reply = asker ? asker(system, user) : std::string();
    if (reply.empty()) {
        DocAttempt a;
        a.document_id = doc.id;
        a.document_text = doc.text;
        a.domain = doc.domain;
        a.failed_stage = "transport";
        a.reason = "the model returned nothing (unreachable, or an empty reply)";
        a.ok = false;
        log.Record(a, escalate);
        res.failed_stage = "transport";
        res.reason = a.reason;
        res.escalated = escalate;
        return res;
    }

    // --- PARSE -------------------------------------------------------------
    std::unique_ptr<Element> tree;
    std::string why;
    if (!ParseElementTree(reply, tree, why)) {
        DocAttempt a;
        a.document_id = doc.id;
        a.document_text = doc.text;
        a.domain = doc.domain;
        a.raw_reply = reply;
        a.failed_stage = "parse";
        a.reason = why;
        a.ok = false;
        log.Record(a, escalate);
        res.failed_stage = "parse";
        res.reason = why;
        res.escalated = escalate;
        return res;
    }

    // --- VALIDATE (the geometry) -------------------------------------------
    // The tree parsed, but does it describe a CLOSED body? A tree with a
    // single sheet is "only one face". This is the check that catches the
    // exact failure the user reported, and it is checkable WITHOUT baking.
    //
    // COUNT THE SHEETS, NOT THE LEAVES: an element that HAS a substance
    // contributes a SHEET even if it also has children. A housing with a
    // rotor inside it has TWO sheets (the housing's ellipsoid and the
    // rotor's ellipsoid), not one -- counting tree-leaves would call that
    // a single sheet and reject a perfectly good closed body.
    size_t sheets = 0;
    {
        std::vector<const Element*> stack{&*tree};
        while (!stack.empty()) {
            const Element* e = stack.back();
            stack.pop_back();
            if (e->substance) ++sheets;
            for (const auto& c : e->children) stack.push_back(c.get());
        }
    }
    if (sheets < 2) {
        // A single sheet is the "only one face" failure. It is rejected
        // because the DOCUMENT path is for reconstructing BODIES: a
        // technical document describes an object with parts, and one sheet
        // cannot be it.
        //
        // (The plain prompt path -- a one-line semantic request -- is a
        // DIFFERENT contract and still allows a single sheet; this check
        // belongs to the DOCUMENT path only, which is why it lives here
        // and not in ParseElementTree.)
        DocAttempt a;
        a.document_id = doc.id;
        a.document_text = doc.text;
        a.domain = doc.domain;
        a.raw_reply = reply;
        a.failed_stage = "validate";
        a.reason = "the tree describes only ONE sheet, so it cannot be a "
                   "closed body (a closed body needs several sheets "
                   "arranged in 3D)";
        a.ok = false;
        log.Record(a, escalate);
        res.failed_stage = "validate";
        res.reason = a.reason;
        res.escalated = escalate;
        return res;
    }

    // --- SUCCESS -----------------------------------------------------------
    DocAttempt a;
    a.document_id = doc.id;
    a.document_text = doc.text;
    a.domain = doc.domain;
    a.raw_reply = reply;
    a.ok = true;
    log.Record(a, escalate);
    res.ok = true;
    res.tree = std::move(tree);
    res.escalated = escalate;
    return res;
}

} // namespace AI
} // namespace DeliRium
