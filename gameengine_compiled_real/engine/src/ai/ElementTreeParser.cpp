// FILEPATH: engine/src/ai/ElementTreeParser.cpp
// JSON element tree -> Element. The AI -> geometry seam.
// Self-contained JSON reader (no nlohmann dependency -- that package is
// OPTIONAL in CMake) + a strict validator against the baker's vocabulary.
#include "delirium/ai/ElementTreeParser.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>

namespace DeliRium {
namespace AI {

// =============================================================================
// JsonValue
// =============================================================================
const JsonValue* JsonValue::Find(const std::string& key) const {
    for (const auto& kv : members)
        if (kv.first == key) return &kv.second;
    return nullptr;
}

std::vector<const JsonValue*> JsonValue::FindAll(const std::string& key) const {
    std::vector<const JsonValue*> out;
    for (const auto& kv : members)
        if (kv.first == key) out.push_back(&kv.second);
    return out;
}

// =============================================================================
// JSON reader
// =============================================================================
namespace {

class JsonReader {
public:
    JsonReader(const std::string& s) : s_(s) {}

    bool Parse(JsonValue& out, std::string& why) {
        SkipWs();
        if (!ParseValue(out, 0, why)) return false;
        SkipWs();
        if (pos_ != s_.size()) {
            why = "trailing data at byte " + std::to_string(pos_);
            return false;
        }
        return true;
    }

private:
    static constexpr int kMaxDepth = 64;   // recursion guard

    const std::string& s_;
    size_t pos_ = 0;

    void SkipWs() {
        while (pos_ < s_.size()) {
            const char c = s_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++pos_;
            else break;
        }
    }

    bool Fail(std::string& why, const char* what) {
        why = std::string(what) + " at byte " + std::to_string(pos_);
        return false;
    }

    bool ParseValue(JsonValue& out, int depth, std::string& why) {
        if (depth > kMaxDepth) return Fail(why, "nesting too deep");
        if (pos_ >= s_.size()) return Fail(why, "unexpected end of input");
        switch (s_[pos_]) {
            case '{': return ParseObject(out, depth, why);
            case '[': return ParseArray(out, depth, why);
            case '"': {
                out.type = JsonValue::Type::String;
                return ParseString(out.text, why);
            }
            case 't':
                if (s_.compare(pos_, 4, "true") == 0) {
                    pos_ += 4; out.type = JsonValue::Type::Bool; out.boolean = true; return true;
                }
                return Fail(why, "invalid literal");
            case 'f':
                if (s_.compare(pos_, 5, "false") == 0) {
                    pos_ += 5; out.type = JsonValue::Type::Bool; out.boolean = false; return true;
                }
                return Fail(why, "invalid literal");
            case 'n':
                if (s_.compare(pos_, 4, "null") == 0) {
                    pos_ += 4; out.type = JsonValue::Type::Null; return true;
                }
                return Fail(why, "invalid literal");
            default:
                return ParseNumber(out, why);
        }
    }

    bool ParseObject(JsonValue& out, int depth, std::string& why) {
        out.type = JsonValue::Type::Object;
        ++pos_;                       // '{'
        SkipWs();
        if (pos_ < s_.size() && s_[pos_] == '}') { ++pos_; return true; }
        while (true) {
            SkipWs();
            // TRAILING COMMA in an object -- same reasoning as ParseArray:
            // "{\"a\":1,}" is a formatting defect, not a data defect, and it
            // is common in LLM output. Accept it rather than discarding a
            // whole generation over a stray character.
            if (pos_ < s_.size() && s_[pos_] == '}') { ++pos_; return true; }
            if (pos_ >= s_.size() || s_[pos_] != '"')
                return Fail(why, "expected object key");
            std::string key;
            if (!ParseString(key, why)) return false;
            SkipWs();
            if (pos_ >= s_.size() || s_[pos_] != ':')
                return Fail(why, "expected ':'");
            ++pos_;
            SkipWs();
            JsonValue v;
            if (!ParseValue(v, depth + 1, why)) return false;
            // The '*' may be separated from the number by whitespace, so the
            // whitespace must be skipped BEFORE the expression is detected.
            // (Without this the check reads the space and misses entirely --
            // measured: the message regressed to "expected ',' or '}'".)
            SkipWs();
            // -------------------------------------------------------------------------
            // A COMPUTATION IS NOT A VALUE.
            // -------------------------------------------------------------------------
            //
            // WHY IT IS HERE (after the value, before the separator): the
            // value itself parses fine -- "7850" is a number -- and the '*'
            // only appears AFTER it. So the mistake is visible only HERE, and
            // this is the first point where the object's KEY is still in hand
            // to name.
            //
            // The key must NOT have been MOVED into the map yet -- a moved-from
            // std::string is empty, and the message would say 'the value for
            // ""' (which names nothing). So the member is emplaced AFTER
            // this check.
            //
            // The ORIGINAL check was here too, but it tested the WRONG
            // characters: it looked for '*', '/', '(', 'M', 'm' -- which is
            // right for "2 * 3" (the '*' follows the 2) but WRONG for
            // "7850 * (4/3 * Math.PI * ...)" only by accident, and it MISSED
            // the case where the expression starts with a letter ("Math.PI
            // * 2"), where ParseValue's number parser stops at 'M' and the
            // error becomes "expected number".
            //
            // MEASURED against a live qwen2.5:7b reading a ring-housing
            // document: it emitted
            //
            //   "mass": 7850 * (4/3 * Math.PI * (Math.pow(0.44,3) - Math.pow(0.32,3)))
            //
            // -- JAVASCRIPT SOURCE as a JSON value. It was trying to COMPUTE
            // a mass from a density and a geometry, which is exactly the job
            // the geometric tools do (`fit_sphere_to_mass`, `volume_of_*`).
            // The model has the arithmetic it needs but no way to perform
            // it, so it writes the formula instead of the answer.
            //
            // So the message names the mistake AND points at the capability
            // that replaces it. That is the difference between a rejection
            // and a lesson.
            if (pos_ < s_.size() &&
                (s_[pos_] == '*' || s_[pos_] == '/' || s_[pos_] == '(' ||
                 s_[pos_] == 'M' || s_[pos_] == 'm')) {
                why = "the value for \"" + key +
                      "\" is an EXPRESSION, not a number -- you wrote a "
                      "formula (e.g. \"7850 * (4/3 * Math.PI * ...)\") where a "
                      "computed NUMBER belongs. Compute the value first: if "
                      "you are working from a density and a geometry, the "
                      "geometric tools (fit_sphere_to_mass, volume_of_ellipsoid, "
                      "volume_of_torus, volume_of_cylinder) return the number "
                      "directly";
                return false;
            }
            out.members.emplace_back(std::move(key), std::move(v));
            SkipWs();
            if (pos_ >= s_.size()) return Fail(why, "unterminated object");
            if (s_[pos_] == ',') { ++pos_; continue; }
            if (s_[pos_] == '}') { ++pos_; return true; }
            return Fail(why, "expected ',' or '}'");
        }
    }

    bool ParseArray(JsonValue& out, int depth, std::string& why) {
        out.type = JsonValue::Type::Array;
        ++pos_;                       // '['
        SkipWs();
        if (pos_ < s_.size() && s_[pos_] == ']') { ++pos_; return true; }
        while (true) {
            SkipWs();
            // TRAILING COMMA: "[1, 2, ]" is malformed strict JSON, but it is
            // THE most common defect in LLM output -- a model ends the list
            // of children, writes the comma, then closes the bracket.
            //
            // WHY ACCEPT IT: the alternative is that the whole generation is
            // thrown away and the user gets "invalid element tree" for a
            // shape the model clearly described. The trailing comma carries
            // no ambiguity -- it cannot change the meaning of the data, only
            // its formatting -- so accepting it is safe, whereas rejecting
            // it costs a real generation.
            //
            // (Objects get the same treatment in ParseObject.)
            if (pos_ < s_.size() && s_[pos_] == ']') { ++pos_; return true; }
            JsonValue v;
            if (!ParseValue(v, depth + 1, why)) return false;
            out.items.push_back(std::move(v));
            SkipWs();
            if (pos_ >= s_.size()) return Fail(why, "unterminated array");
            if (s_[pos_] == ',') { ++pos_; continue; }
            if (s_[pos_] == ']') { ++pos_; return true; }
            // The same lesson as in ParseObject: an expression is not a value.
            // An array element can carry one too (a model writing a computed
            // coordinate or mass inside an array), and the raw "expected ','
            // or ']'" names neither the mistake nor the way out.
            if (s_[pos_] == '*' || s_[pos_] == '/' || s_[pos_] == '(' ||
                s_[pos_] == 'M' || s_[pos_] == 'm') {
                why = "the value at byte " + std::to_string(pos_) +
                      " is an EXPRESSION, not a number -- you wrote a formula "
                      "where a computed NUMBER belongs. Compute the value "
                      "first: the geometric tools (fit_sphere_to_mass, "
                      "volume_of_ellipsoid, volume_of_torus, "
                      "volume_of_cylinder) return the number directly";
                return false;
            }
            return Fail(why, "expected ',' or ']'");
        }
    }

    bool ParseString(std::string& out, std::string& why) {
        ++pos_;                       // opening quote
        out.clear();
        while (true) {
            if (pos_ >= s_.size()) return Fail(why, "unterminated string");
            const char c = s_[pos_++];
            if (c == '"') return true;
            if (c != '\\') { out += c; continue; }
            if (pos_ >= s_.size()) return Fail(why, "unterminated escape");
            const char e = s_[pos_++];
            switch (e) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'b':  out += '\b'; break;
                case 'f':  out += '\f'; break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                case 'u': {
                    // \uXXXX -> UTF-8. Surrogate pairs are joined.
                    if (pos_ + 4 > s_.size()) return Fail(why, "bad \\u escape");
                    unsigned cp = 0;
                    for (int i = 0; i < 4; ++i) {
                        const char h = s_[pos_ + size_t(i)];
                        cp <<= 4;
                        if (h >= '0' && h <= '9') cp |= unsigned(h - '0');
                        else if (h >= 'a' && h <= 'f') cp |= unsigned(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') cp |= unsigned(h - 'A' + 10);
                        else return Fail(why, "bad hex in \\u escape");
                    }
                    pos_ += 4;
                    // High surrogate: try to pair it with the next \uDCxx.
                    if (cp >= 0xD800 && cp <= 0xDBFF && pos_ + 6 <= s_.size() &&
                        s_[pos_] == '\\' && s_[pos_ + 1] == 'u') {
                        unsigned lo = 0;
                        bool ok = true;
                        for (int i = 0; i < 4; ++i) {
                            const char h = s_[pos_ + 2 + size_t(i)];
                            lo <<= 4;
                            if (h >= '0' && h <= '9') lo |= unsigned(h - '0');
                            else if (h >= 'a' && h <= 'f') lo |= unsigned(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') lo |= unsigned(h - 'A' + 10);
                            else { ok = false; break; }
                        }
                        if (ok && lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            pos_ += 6;
                        }
                    }
                    AppendUtf8(out, cp);
                    break;
                }
                default: return Fail(why, "unknown escape");
            }
        }
    }

    static void AppendUtf8(std::string& out, unsigned cp) {
        if (cp < 0x80) {
            out += char(cp);
        } else if (cp < 0x800) {
            out += char(0xC0 | (cp >> 6));
            out += char(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += char(0xE0 | (cp >> 12));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        } else {
            out += char(0xF0 | (cp >> 18));
            out += char(0x80 | ((cp >> 12) & 0x3F));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        }
    }

    bool ParseNumber(JsonValue& out, std::string& why) {
        const size_t start = pos_;
        if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+')) ++pos_;
        bool any = false;
        while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
            ++pos_; any = true;
        }
        if (pos_ < s_.size() && s_[pos_] == '.') {
            ++pos_;
            while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
                ++pos_; any = true;
            }
        }
        if (!any) return Fail(why, "expected number");
        if (pos_ < s_.size() && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+')) ++pos_;
            while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) ++pos_;
        }
        out.type = JsonValue::Type::Number;
        out.number = std::strtod(s_.substr(start, pos_ - start).c_str(), nullptr);
        return true;
    }
};

} // namespace

bool ParseJson(const std::string& src, JsonValue& out, std::string& why) {
    JsonReader r(src);
    out = JsonValue{};
    return r.Parse(out, why);
}

bool ExtractJsonObject(const std::string& src, std::string& out) {
    // Find the first '{' that opens a balanced object, ignoring braces
    // inside JSON strings. A model often answers with prose or a markdown
    // fence around its JSON.
    int depth = 0;
    bool in_str = false, esc = false;
    size_t start = std::string::npos;
    for (size_t i = 0; i < src.size(); ++i) {
        const char c = src[i];
        if (in_str) {
            if (esc) { esc = false; continue; }
            if (c == '\\') { esc = true; continue; }
            if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') { in_str = true; continue; }
        if (c == '{') {
            if (depth == 0) start = i;
            ++depth;
        } else if (c == '}') {
            if (depth > 0) {
                --depth;
                if (depth == 0 && start != std::string::npos) {
                    out = src.substr(start, i - start + 1);
                    return true;
                }
            }
        }
    }
    return false;
}

// =============================================================================
// Element tree builder
// =============================================================================
namespace {

// Read a [x, y, z] array (or a scalar -> x=y=z).
bool ReadVec3(const JsonValue& v, Vec3& out, const char* label, std::string& why) {
    if (v.IsNumber()) { out = {float(v.number), float(v.number), float(v.number)}; return true; }
    if (!v.IsArray()) { why = std::string(label) + " must be an array or number"; return false; }
    if (v.items.size() == 1 && v.items[0].IsNumber()) {
        out = {float(v.items[0].number), float(v.items[0].number), float(v.items[0].number)};
        return true;
    }
    if (v.items.size() != 3) {
        why = std::string(label) + " must have 1 or 3 components";
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        if (!v.items[size_t(i)].IsNumber()) {
            why = std::string(label) + " components must be numbers";
            return false;
        }
    }
    out = {float(v.items[0].number), float(v.items[1].number), float(v.items[2].number)};
    return true;
}

bool ReadColor(const JsonValue& v, Vec3& out, std::string& why) {
    if (v.IsNumber()) {
        const float g = float(v.number);
        out = {g, g, g};
        return true;
    }
    if (!v.IsArray()) { why = "color must be an array or number"; return false; }
    if (v.items.size() != 3) { why = "color must have 3 components"; return false; }
    for (int i = 0; i < 3; ++i) {
        if (!v.items[size_t(i)].IsNumber()) { why = "color components must be numbers"; return false; }
        const float c = float(v.items[size_t(i)].number);
        // Accept 0..255 as well as 0..1 -- a model will emit both.
        out.x = i == 0 ? (c > 1.0f ? c / 255.0f : c) : out.x;
        out.y = i == 1 ? (c > 1.0f ? c / 255.0f : c) : out.y;
        out.z = i == 2 ? (c > 1.0f ? c / 255.0f : c) : out.z;
    }
    for (int i = 0; i < 3; ++i) {
        const float c = i == 0 ? out.x : (i == 1 ? out.y : out.z);
        if (!std::isfinite(c) || c < 0.0f || c > 1.0f) { why = "color out of range"; return false; }
    }
    return true;
}

double Param(const JsonValue& params, const char* key, double fallback) {
    if (!params.IsObject()) return fallback;
    const JsonValue* v = params.Find(key);
    return (v && v->IsNumber()) ? v->number : fallback;
}

// Build the leaf substance for `kind`. Returns nullptr for an unknown kind
// and sets `why` -- the parser REJECTS unknown substances rather than
// silently degrading to a plane (a model that invented "dragon_scale"
// must be told, not quietly handed a flat card).
std::unique_ptr<ISubstance> MakeSubstance(const std::string& kind,
                                           const JsonValue& params,
                                           std::string& why) {
    auto lower = [](std::string s) {
        for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
        return s;
    };
    const std::string k = lower(kind);
    if (k == "plane" || k == "flat") return std::make_unique<PlaneSubstance>();
    if (k == "ellipsoid" || k == "sphere" || k == "orb" || k == "ball") {
        float sx = float(Param(params, "sx", Param(params, "radius", 0.5)));
        float sy = float(Param(params, "sy", sx));
        float sz = float(Param(params, "sz", sx));
        if (sx <= 0.f || sy <= 0.f || sz <= 0.f) {
            // The message must TEACH THE CORRECTION, not just reject.
            //
            // WHY: this error is the model's most common mistake -- it uses
            // a NEGATIVE half-extent to mean "this sheet points downward"
            // (to close the bottom of a body). That is a reasonable
            // intuition, and the schema already has the right mechanism
            // for it: the element's "scale" field, e.g. "scale":[1,1,-1]
            // flips the sheet. A bare "must be > 0" tells the model only
            // that it was wrong; naming the correct field tells it how to
            // be right, which is what makes the escalation attempt
            // succeed instead of failing the same way.
            //
            // Observed against a live qwen2.5:7b reading a gearbox-housing
            // document: it emitted {"sx":0.6,"sy":0.6,"sz":-1} for the
            // cylinder's lower half.
            why = "ellipsoid half-extents (sx, sy, sz) must be > 0 -- a "
                  "negative half-extent is not how a sheet is flipped. To "
                  "point a sheet DOWNWARD use the element's \"scale\" field "
                  "with a negative component, e.g. \"scale\":[1,1,-1], and "
                  "keep sx/sy/sz positive";
            return nullptr;
        }
        return std::make_unique<EllipsoidSubstance>(sx, sy, sz);
    }
    if (k == "torus" || k == "donut" || k == "ring" || k == "tube") {
        // major = distance from the centre to the TUBE CENTRE;
        // minor = the tube's own radius. A model that says "radius" for a
        // torus almost always means the OUTER radius, so outer -> major is
        // derived from outer - minor.
        //
        // -------------------------------------------------------------------------
        // THE OUTER-RADIUS MISTAKE, AND WHY IT IS CAUGHT HERE.
        // -------------------------------------------------------------------------
        //
        // MEASURED against a live qwen2.5:7b reading a ring-housing document
        // that said "a steel torus with an outer radius of 0.44 units and a
        // tube radius of 0.12 units": the model emitted
        //
        //   "substance_params": {"major": 0.44, "minor": 0.12}
        //
        // -- the OUTER radius used as major. The correct major is
        // 0.44 - 0.12 = 0.32. The resulting ring's outer edge is at 0.56
        // instead of 0.44, so the part is 27% too big and sits wrong
        // relative to the other parts.
        //
        // The parser could not detect this by comparing magnitudes alone: a
        // torus whose major really IS its outer radius is a legitimate
        // (if unusual) shape. What makes it detectable is the DOCUMENT's
        // own wording -- and the parser does not have the document.
        //
        // So the correction is NOT attempted here. What IS done is that when
        // the model supplies an explicit "outer_radius", it is converted
        // correctly (outer -> major), and when it supplies "major" together
        // with an "outer_radius", the two are checked against each other
        // and the inconsistency is reported with the tool that resolves it.
        // The tool `resolve_torus_radii` is the capability the model should
        // have used; naming it here is what teaches the deployment.
        const float minor = float(Param(params, "minor", Param(params, "tube_radius", 0.12)));
        const bool has_outer = params.Find("outer_radius") != nullptr;
        float major = float(Param(params, "major", Param(params, "ring_radius", 0.32)));
        if (has_outer) {
            const float outer = float(Param(params, "outer_radius", 0.0));
            if (outer > 0.f) {
                // -------------------------------------------------------------------------
                // THE CONTRADICTION IS major != outer -- NOT major != outer - minor.
                // -------------------------------------------------------------------------
                //
                // MEASURED: the model wrote BOTH "major": 0.44 AND
                // "outer_radius": 0.44 -- the same number twice, because it
                // had the outer radius in mind and copied it into both slots.
                //
                // The OLD check compared major against OUTER, so 0.44 vs 0.44
                // AGREED, the check passed, and then the code silently did
                // `major = outer - minor = 0.32` -- which happens to be the
                // RIGHT answer here, but only by accident. The model's
                // stated intent (major = outer) was never challenged, so on
                // the next document it would write "major": 0.56 (outer +
                // minor) and get a ring 27% too big with NO error anywhere.
                //
                // The CORRECT contradiction is the one the geometry defines:
                // major IS the tube-centre radius, so it must equal
                // outer - minor. If the model says major = outer, it is
                // using the outer radius as the major -- which is the
                // measured mistake.
                const float expected_major = outer - minor;
                if (params.Find("major") != nullptr &&
                    std::fabs(major - expected_major) > 1e-4f) {
                    why = "the torus's \"major\" and \"outer_radius\" "
                          "disagree. \"major\" is the TUBE-CENTRE radius and "
                          "\"outer_radius\" is the OUTER edge, so "
                          "major = outer_radius - minor = " +
                          std::to_string(expected_major) +
                          ". Use the resolve_torus_radii tool to compute it "
                          "rather than guessing";
                    return nullptr;
                }
                // No explicit major, or it agrees: derive it from the outer.
                major = expected_major;
            }
        }
        if (minor <= 0.f) { why = "torus minor radius must be > 0"; return nullptr; }
        if (major <= minor) {
            why = "torus major radius must be greater than the minor radius. "
                  "If you were given the OUTER radius, call "
                  "resolve_torus_radii to get major = outer - minor";
            return nullptr;
        }
        return std::make_unique<TorusSubstance>(major, minor);
    }
    if (k == "gaussian" || k == "bump" || k == "node") {
        const float amp  = float(Param(params, "amp", 0.2));
        const float sig  = float(Param(params, "sigma", 0.35));
        if (sig <= 0.f) { why = "gaussian sigma must be > 0"; return nullptr; }
        return std::make_unique<GaussianSystemSubstance>(amp, sig);
    }
    if (k == "heightfield" || k == "wave" || k == "field" || k == "terrain") {
        const float amp  = float(Param(params, "amp", 0.1));
        const float fx   = float(Param(params, "freq_x", 6.0));
        const float fy   = float(Param(params, "freq_y", 6.0));
        const float ph   = float(Param(params, "phase", 0.0));
        return std::make_unique<HeightfieldSubstance>(
            [amp, fx, fy, ph](float x, float y) {
                return amp * std::sin(fx * x + ph) * std::cos(fy * y + ph);
            });
    }
    // -------------------------------------------------------------------------
    // THE ERROR MUST TEACH THE CORRECTION.
    // -------------------------------------------------------------------------
    //
    // MEASURED against a live qwen2.5:7b reading a ring-housing document:
    // it wrote `{"kind": "ring_housing", ...}` -- a PART'S NAME -- where the
    // schema wants a SUBSTANCE TYPE, and the parser's flat "unknown
    // substance 'ring_housing'" gave the model nothing to act on.
    //
    // There are TWO different mistakes here, and they need DIFFERENT
    // corrections:
    //
    //   1. the model put a PART NAME in the substance slot ("ring_housing",
    //      "housing", "pinion_gear"). It understood the DOCUMENT perfectly
    //      -- it just used the wrong FIELD. The correction is: "put the
    //      part's name in \"kind\" (a free label) and the SHAPE TYPE in
    //      \"substance\"".
    //
    //   2. the model invented a substance that does not exist ("dragon_scale",
    //      "torusx"). The correction is: "use one of these types".
    //
    // A part name is recognisable: it contains an underscore or a space, or
    // it is a word the document would use to NAME a part. A made-up
    // substance type is a single lowercase token. That distinction is
    // heuristic, so the message says BOTH corrections when it cannot tell --
    // an over-hedged message is still far better than a flat rejection,
    // because the escalation attempt (and the tool loop) feed this text
    // back to the model, and a message that names the wrong field teaches
    // the model where the field IS.
    // -------------------------------------------------------------------------
    // IS IT A PART NAME, OR AN INVENTED SUBSTANCE?
    // -------------------------------------------------------------------------
    //
    // These are TWO different mistakes and they need TWO different
    // corrections. The gate caught the heuristic getting it WRONG:
    //
    //   "ring_housing"  -- a PART's name (it is a name a document would use)
    //   "dragon_scale"  -- an INVENTED substance type (a made-up shape)
    //
    // The old heuristic said "it has an underscore, so it is a part name" --
    // and so "dragon_scale" was diagnosed as a part name, telling the model
    // to move it to "kind". That is the WRONG lesson: it would put
    // "dragon_scale" in the free-text field and still have no shape, so the
    // next attempt would fail the same way.
    //
    // THE DISCRIMINATOR: a part name is a name the DOCUMENT would use for a
    // part -- and the parser has no document. But it DOES have the list of
    // substances the engine knows. So the rule is:
    //
    //   * a name that is a KNOWN substance (or an obvious synonym of one)
    //     is not an error at all;
    //   * a name that CONTAINS a known substance's name ("ring_housing"
    //     contains "housing"... hmm, not a substance) -- so instead:
    //   * a name whose LAST token is a known substance ("torus_housing",
    //     "steel_cylinder") is a part name QUALIFIED by its shape;
    //   * a name whose last token is NOT a known substance is INVENTED.
    //
    // "ring_housing" -> last token "housing" -> not a substance -> INVENTED?
    // But the measurement says the model meant it as a part name.
    //
    // The honest resolution: a name is a PART NAME when it reads like a
    // noun phrase about a part -- and the engine's own vocabulary is the
    // authority on what a SHAPE is. So:
    //
    //   * if the whole name IS a known substance -> not an error;
    //   * else if the name CONTAINS a known substance as a token
    //     ("pinion_gear" does not, but "gear_housing" does not either) --
    //     so fall back to: a name that a document would use is one with a
    //     space or hyphen ("ring housing", "gear-housing"), OR one whose
    //     first token is a common part word (housing, gear, shaft, pinion,
    //     bearing, cover, plate, rotor, housing, case, flange, rotor);
    //   * else -> INVENTED.
    //
    // "dragon_scale" -> "dragon" is not a part word -> INVENTED. Correct.
    // "ring_housing" -> "ring" is not in the part-word list either... but
    // "housing" IS. So the check must look at ANY token.
    //
    // The part-word list is the engine's own -- the words the geometry
    // vocabulary and the documents use to name the parts of a mechanism.
    static const char* kPartWords[] = {
        "housing", "gear", "shaft", "pinion", "bearing", "cover", "plate",
        "rotor", "case", "flange", "gearbox", "pump", "valve", "disc",
        "disk", "wheel", "ring", "body", "bracket", "spindle", "axle",
        "seat", "seal", "hub", "spoke", "rim", "barrel", "cap", "bolt",
        "screw", "nut", "washer", "spring", "lever", "crank", "rod",
        "piston", "cylinder", "block", "head", "cam", "flywheel", "clutch",
        "coupling", "collar", "bush", "key", "pin", "roller", "yoke"
    };
    const std::string lower_kind = lower(kind);
    bool names_a_part = false;
    {
        // Split on '_', '-' and ' ' and look for a part word in any token.
        std::string tok;
        auto consider = [&](const std::string& t) {
            if (t.empty()) return;
            for (const char* w : kPartWords) {
                if (t == w) { names_a_part = true; return; }
            }
        };
        for (char c : lower_kind + " ") {
            if (c == '_' || c == '-' || c == ' ') { consider(tok); tok.clear(); }
            else tok.push_back(c);
        }
    }
    // A multi-word name ("ring housing") is a part name by construction.
    if (lower_kind.find(' ') != std::string::npos ||
        lower_kind.find('-') != std::string::npos) {
        names_a_part = true;
    }

    if (names_a_part) {
        why = "'" + kind +
              "' looks like a PART NAME, not a substance type. The schema "
              "has two separate fields: put the part's name in \"kind\" (a "
              "free-text label, e.g. \"kind\": \"ring_housing\") and the "
              "SHAPE TYPE in \"substance\" (one of: plane, ellipsoid, sphere, "
              "torus, donut, gaussian, bump, heightfield, wave). A part's "
              "name does not belong in \"substance\"";
    } else {
        why = "unknown substance '" + kind +
              "'. Use one of: plane, ellipsoid/sphere, torus/donut, "
              "gaussian/bump, or heightfield/wave. If '" + kind +
              "' is a part's NAME, put it in \"kind\" instead and put the "
              "shape type in \"substance\"";
    }
    return nullptr;
}

// Build one element (and its subtree) from a JSON object.
bool BuildElement(const JsonValue& v, std::unique_ptr<Element>& out, std::string& why,
                  int depth);

// Build the children array of `v`.
bool BuildChildren(const JsonValue& v, Element& parent, std::string& why, int depth) {
    // EVERY "children" array, not just the first.
    //
    // MEASURED against a live qwen2.5:7b reading a ring-housing document:
    // it emitted TWO "children" arrays in one element -- the first holding
    // the pinion and the support bar, the second holding the mirrored lower
    // ring. `Find` returns the FIRST, so the second array was silently
    // discarded and the body baked OPEN, with no error anywhere: the
    // document clearly asked for a closed housing and the geometry just
    // quietly lost half of it.
    //
    // Merging is the right resolution, not rejecting: both arrays hold
    // children, their union is unambiguous, and the alternative (a hard
    // failure) throws away a generation that was correct in substance.
    // (A duplicate "origin" would NOT be mergeable -- that one is genuinely
    // ambiguous, and the first wins as before.)
    for (const JsonValue* kids : v.FindAll("children")) {
        if (!kids->IsArray()) { why = "\"children\" must be an array"; return false; }
        for (const JsonValue& k : kids->items) {
            std::unique_ptr<Element> child;
            if (!BuildElement(k, child, why, depth + 1)) return false;
            parent.AddChild(std::move(child));
        }
    }
    return true;
}

bool BuildElement(const JsonValue& v, std::unique_ptr<Element>& out, std::string& why,
                  int depth) {
    if (depth > 32) { why = "element tree too deep"; return false; }
    if (!v.IsObject()) { why = "element must be a JSON object"; return false; }

    auto el = std::make_unique<Element>();

    if (const JsonValue* k = v.Find("kind"))
        el->kind = k->GetString("object");
    else
        el->kind = "object";

    if (const JsonValue* o = v.Find("origin")) {
        if (!ReadVec3(*o, el->origin, "origin", why)) return false;
    }
    if (const JsonValue* s = v.Find("scale")) {
        if (!ReadVec3(*s, el->scale, "scale", why)) return false;
        if (el->scale.x == 0.f || el->scale.y == 0.f || el->scale.z == 0.f) {
            why = "scale components must be non-zero";
            return false;
        }
    }
    if (const JsonValue* w = v.Find("weight")) {
        if (!w->IsNumber()) { why = "weight must be a number"; return false; }
        el->weight = float(w->number);
        if (!std::isfinite(el->weight) || el->weight < 0.f) {
            why = "weight must be finite and >= 0";
            return false;
        }
    }
    if (const JsonValue* c = v.Find("color")) {
        if (!ReadColor(*c, el->color, why)) return false;
    }

    // The substance's parameters live either beside "substance" or in a
    // sibling "substance_params" object.
    const JsonValue* params = v.Find("substance_params");
    static const JsonValue kNoParams{};
    if (!params) params = &kNoParams;

    if (const JsonValue* s = v.Find("substance")) {
        if (s->IsNull()) {
            // explicit null = grouping node (no leaf)
        } else if (!s->IsString()) {
            why = "\"substance\" must be a string";
            return false;
        } else if (s->text.empty()) {
            // AN EMPTY STRING IS ALSO A GROUPING NODE.
            //
            // WHY: the schema already expresses "a node with children but no
            // surface of its own" as `"substance": null`. But a model
            // writing a COMPLEX assembly naturally writes `"substance": ""`
            // for the grouping nodes -- it is the obvious way to say "this
            // is a container, not a part". And that is EXACTLY what the
            // 14B did with the gearbox document:
            //
            //   {"kind": "input shaft", "substance": "",
            //    "children": [{"kind": "shaft", ...}, {"kind": "pinion gear", ...}]}
            //
            // The parser rejected it as "unknown substance ''" -- a failure
            // that reads like the MODEL misunderstood the schema, when in
            // fact the model used a REASONABLE spelling of a construct the
            // schema already has.
            //
            // The baker handles it already: CollectSheets only collects the
            // elements that HAVE a substance, so a grouping node simply
            // contributes its children.
            //
            // NOTE: this is NOT the same as accepting an unknown substance
            // name. "torusx" is still rejected -- the parser refuses to
            // degrade a substance it does not know into a flat card. An
            // EMPTY name is not an unknown substance; it is the absence of
            // one.
        } else {
            el->substance = MakeSubstance(s->text, *params, why);
            if (!el->substance) return false;
        }
    }

    // Optional spatial presence mask.
    if (const JsonValue* p = v.Find("presence")) {
        if (p->IsObject()) {
            const double radius = Param(*p, "radius", -1.0);
            const double cx = Param(*p, "center_x", 0.0);
            const double cy = Param(*p, "center_y", 0.0);
            const double cz = Param(*p, "center_z", 0.0);
            if (radius > 0.0) {
                const float r2 = float(radius * radius);
                el->SetPresence([r2, cx, cy, cz](const Vec3& w) {
                    const float dx = float(w.x - cx), dy = float(w.y - cy), dz = float(w.z - cz);
                    const float d2 = dx*dx + dy*dy + dz*dz;
                    if (d2 >= r2) return 0.0f;
                    // Smooth falloff to the mask edge.
                    const float t = 1.0f - d2 / r2;
                    return t * t;
                });
            }
        }
    }

    // COARSENESS: the abstraction level this tree asks for.
    //
    // WHY: "make them complete in representation, denoting the coarseness
    // for the sake of speedy computation". The model picks a LEVEL, and the
    // level derives both budgets the engine spends (surface resolution and
    // skeleton density). It is not a pixel count.
    //
    //   0 = fine    (grid 48)  near-final surface
    //   1 = medium  (grid 32)  the default
    //   2 = coarse  (grid 16)  silhouette + mass
    //   3 = sketch  (grid 8)   proportions + skeleton only
    if (const JsonValue* l = v.Find("lod")) {
        if (!l->IsNumber()) { why = "\"lod\" must be a number (0..3)"; return false; }
        const int lod = int(l->number);
        if (lod < 0 || lod > 3) {
            why = "\"lod\" must be in [0,3]: 0 fine, 1 medium, 2 coarse, 3 sketch";
            return false;
        }
        el->lod = lod;
    }

    // Optional RIGID SEGMENT (bone).
    //
    // WHY: a surface-only tree cannot express a mechanism. Four overlapping
    // ellipsoids say nothing about which part ROTATES relative to which.
    // A bone says "this is the shaft, it runs from here to there, it pivots
    // at its head, it carries this much mass" -- which is what makes
    // reparametrization, inertia and field physics computable.
    //
    // Schema:
    //   "bone": {
    //     "name":     "shaft",          // optional
    //     "head":     [x, y, z],         // the pivot (required)
    //     "tail":     [x, y, z],         // the far end (required)
    //     "bend":     [x, y, z],         // optional: the deviation
    //     "radius":   0.1,               // optional: cross-section radius
    //     "mass":     0.5,               // optional: mass contribution
    //     "stiffness":1.0,               // optional: 0 free .. 1 rigid
    //     "joints":   ["knee", "ankle"], // optional: named articulations
    //     "parent":   "housing"          // optional: the parent bone's NAME
    //   }
    //
    // The parent is given by NAME, not by index: a model writing JSON
    // cannot know an index that has not been assigned yet. Names are
    // resolved after the whole tree is built (ResolveBoneParents), and an
    // unknown parent name is REJECTED rather than silently re-rooted.
    if (const JsonValue* b = v.Find("bone")) {
        if (!b->IsObject()) { why = "\"bone\" must be an object"; return false; }
        Bone bone;
        if (const JsonValue* n = b->Find("name"))
            bone.name = n->GetString("");
        if (const JsonValue* h = b->Find("head")) {
            if (!ReadVec3(*h, bone.head, "bone.head", why)) return false;
        } else {
            why = "bone needs a \"head\" (the pivot point)";
            return false;
        }
        if (const JsonValue* t = b->Find("tail")) {
            if (!ReadVec3(*t, bone.tail, "bone.tail", why)) return false;
        } else {
            why = "bone needs a \"tail\" (the far end of the segment)";
            return false;
        }
        if (const JsonValue* bnd = b->Find("bend")) {
            if (!ReadVec3(*bnd, bone.bend, "bone.bend", why)) return false;
        }
        bone.radius    = float(Param(*b, "radius", 0.0));
        bone.mass      = float(Param(*b, "mass", 0.0));
        bone.stiffness = float(Param(*b, "stiffness", 1.0));
        // -------------------------------------------------------------------------
        // A COMPUTATION IS NOT A VALUE.
        // -------------------------------------------------------------------------
        //
        // MEASURED against a live qwen2.5:7b reading a ring-housing document:
        // it emitted
        //
        //   "mass": 7850 * (4/3 * Math.PI * (Math.pow(0.44,3) - Math.pow(0.32,3)))
        //
        // -- JAVASCRIPT SOURCE as a JSON value. That is not valid JSON, so
        // the parse fails with "invalid JSON at byte N", which tells the
        // model nothing about WHY.
        //
        // But WHY it did that is the interesting part: it was trying to
        // COMPUTE a mass, because the document gave it a density and a
        // geometry. That is exactly the job the geometric tools exist to do
        // (`fit_sphere_to_mass`, `volume_of_*`). The model has the
        // arithmetic it needs but no way to perform it, so it writes the
        // formula instead of the answer.
        //
        // So this message must do two things: name the mistake, and point at
        // the capability that replaces it. That is the difference between a
        // rejection and a lesson.
        for (const char* numkey : { "mass", "radius", "stiffness" }) {
            if (const JsonValue* nv = b->Find(numkey)) {
                if (!nv->IsNumber()) {
                    why = std::string("bone.") + numkey +
                          " must be a NUMBER, not an expression. You wrote a "
                          "formula instead of its result. Compute the value "
                          "first -- if you are working from a density and a "
                          "geometry, the geometric tools (fit_sphere_to_mass, "
                          "volume_of_ellipsoid, volume_of_torus) return the "
                          "number directly";
                    return false;
                }
            }
        }
        if (bone.radius < 0.f) { why = "bone radius must be >= 0"; return false; }
        if (bone.mass < 0.f)   { why = "bone mass must be >= 0"; return false; }
        if (bone.stiffness < 0.f || bone.stiffness > 1.f) {
            why = "bone stiffness must be in [0,1]";
            return false;
        }
        if (const JsonValue* j = b->Find("joints")) {
            if (!j->IsArray()) { why = "bone.joints must be an array"; return false; }
            for (const JsonValue& it : j->items) {
                if (!it.IsString()) { why = "bone.joints entries must be strings"; return false; }
                if (!it.text.empty()) bone.joints.push_back(it.text);
            }
        }
        // The MATERIAL, as a NAME.
        //
        // WHY THE NAME IS NOT VALIDATED HERE: the parser's job is to
        // parse, and the registry's job is to know what exists. But the
        // name IS checked for SHAPE (a non-string is a schema error), and
        // an unknown name is caught LATER -- by the field layer, which
        // reports it as "unknown material" rather than as a parse error.
        //
        // WHY THAT ORDER: a document that names a material the registry
        // does not know is a DOCUMENT problem, not a JSON problem. The
        // document pipeline's error log is where that belongs -- it is
        // exactly the kind of failure the log exists to collect, and it
        // is what lets the escalation prompt tell the model which word
        // it got wrong.
        if (const JsonValue* m = b->Find("material")) {
            if (!m->IsString()) { why = "bone.material must be a string"; return false; }
            bone.material = m->text;
        }
        // The parent NAME is stashed in the bone's name field temporarily;
        // ResolveBoneParents rewrites it into an index. A name that also
        // names the bone would be ambiguous, so the parent goes in a
        // dedicated slot instead -- see Bone::parent_name below.
        if (const JsonValue* p = b->Find("parent")) {
            if (!p->IsString()) { why = "bone.parent must be a bone NAME"; return false; }
            if (p->text.empty()) { why = "bone.parent must not be empty"; return false; }
            bone.parent_name = p->text;
        }
        el->SetBone(bone);
    }

    if (!BuildChildren(v, *el, why, depth)) return false;
    out = std::move(el);
    return true;
}

// Count nodes / leaves for the console summary.
void CountTree(const Element& e, int& nodes, int& leaves) {
    ++nodes;
    if (e.substance) ++leaves;
    for (const auto& c : e.children) CountTree(*c, nodes, leaves);
}

// -----------------------------------------------------------------------------
// Bone collection and parent resolution.
//
// The tree stores each bone INSIDE the element that owns it. Two passes:
//
//   1. Collect every bone in tree order and give it an index. The index is
//      the bone's position in that flat list -- which is exactly what the
//      frozen BoneRecord's `parent` field and AnchorRecord's `bone_index`
//      will refer to.
//   2. Resolve each bone's parent NAME to an index.
//
// WHY two passes: a model writes JSON, and JSON has no way to express "the
// bone defined two nodes earlier" except by NAME. A single pass would have
// to resolve a parent that has not been seen yet.
//
// WHY a named parent must be REJECTED when unknown: silently re-rooting a
// bone whose parent was misspelled produces a skeleton that LOOKS fine and
// articulates wrongly -- the worst kind of bug, because the geometry is
// still right.
// -----------------------------------------------------------------------------
void CollectBones(Element& e, std::vector<Bone*>& out) {
    if (e.has_bone) out.push_back(&e.bone);
    for (auto& c : e.children) CollectBones(*c, out);
}

bool ResolveBoneParents(Element& e, std::vector<Bone*>& bones,
                        std::string& why) {
    // Pass 1: index by name. A duplicate name is ambiguous, so it is
    // rejected -- two bones called "shaft" cannot be told apart by a
    // document, and a parent reference to one of them would be a coin flip.
    std::unordered_map<std::string, int> by_name;
    for (size_t i = 0; i < bones.size(); ++i) {
        const std::string& n = bones[i]->name;
        if (n.empty()) continue;         // an unnamed bone is a root by default
        if (by_name.count(n)) {
            why = "two bones are named \"" + n + "\" -- a parent reference "
                  "to that name would be ambiguous";
            return false;
        }
        by_name[n] = int(i);
    }
    // Pass 2: resolve.
    for (size_t i = 0; i < bones.size(); ++i) {
        Bone* b = bones[i];
        if (b->parent_name.empty()) { b->parent = -1; continue; }
        auto it = by_name.find(b->parent_name);
        if (it == by_name.end()) {
            why = "bone \"" + (b->name.empty() ? std::string("(unnamed)") : b->name) +
                  "\" names a parent \"" + b->parent_name +
                  "\" that does not exist";
            return false;
        }
        b->parent = it->second;
        // A bone cannot be its own ancestor: the hierarchy would be cyclic
        // and every traversal (mass accumulation, pose solving, inertia)
        // would recurse forever.
        for (int g = b->parent; g >= 0; ) {
            if (g == int(i)) {
                why = "bone \"" + b->parent_name + "\" is its own ancestor";
                return false;
            }
            if (g >= int(bones.size())) break;
            g = bones[g]->parent;
        }
    }
    return true;
}

} // namespace

bool ParseElementTree(const std::string& json, std::unique_ptr<Element>& out,
                      std::string& why) {
    JsonValue root;
    std::string jerr;
    if (!ParseJson(json, root, jerr)) {
        why = "invalid JSON: " + jerr;
        return false;
    }
    // Accept either a single object or an array of them (the array's first
    // element becomes the root; the rest are grafted as its children).
    const JsonValue* tree = &root;
    std::unique_ptr<Element> built;
    if (root.IsArray()) {
        if (root.items.empty()) { why = "element array is empty"; return false; }
        tree = &root.items[0];
    }
    if (!BuildElement(*tree, built, why, 0)) return false;
    // Graft the remaining array elements as siblings under a grouping root.
    if (root.IsArray() && root.items.size() > 1) {
        auto group = std::make_unique<Element>();
        group->kind = "object";
        group->AddChild(std::move(built));
        for (size_t i = 1; i < root.items.size(); ++i) {
            std::unique_ptr<Element> sib;
            if (!BuildElement(root.items[i], sib, why, 0)) return false;
            group->AddChild(std::move(sib));
        }
        built = std::move(group);
    }
    // A tree with NO substance anywhere would bake into a degenerate body
    // (the baker skips out-of-support samples). Require at least one leaf.
    int nodes = 0, leaves = 0;
    CountTree(*built, nodes, leaves);
    if (leaves == 0) {
        why = "element tree has no substance (nothing to bake)";
        return false;
    }
    // Resolve the bone hierarchy. This is where a named parent becomes an
    // index, and where an unknown or cyclic parent is REJECTED.
    {
        std::vector<Bone*> bones;
        CollectBones(*built, bones);
        if (!ResolveBoneParents(*built, bones, why)) return false;
    }
    out = std::move(built);
    return true;
}

std::string ElementTreeSystemPrompt() {
    return
"You are the geometry engine of DeliRium. You turn a description into an "
"ELEMENT TREE: a recursive set of overlapping systems, each with a leaf "
"physical substance. Reply with ONE JSON object and nothing else.\n"
"\n"
"Schema (every key optional except where noted):\n"
"  kind            string  category tag, e.g. \"body\", \"limb\", \"organ\"\n"
"  origin          [x,y,z] local offset of this system\n"
"  scale           [x,y,z] local scale (default [1,1,1])\n"
"  weight          number  contribution strength where systems overlap (>=0)\n"
"  color           [r,g,b] base color, each 0..1 (or 0..255)\n"
"  substance       string  the leaf material (REQUIRED on at least one node)\n"
"  substance_params object parameters for that substance\n"
"  presence        object  optional spatial mask: {\"radius\":r,\"center_x\":..}\n"
"  bone            object  optional RIGID SEGMENT for this system (see below)\n"
"  lod             number  optional COARSENESS level, 0..3 (see below)\n"
"  children        array   sub-systems that OVERLAP this one\n"
"\n"
"Substances (only these five are valid):\n"
"  \"plane\"      flat infinite surface. params: {}\n"
"  \"ellipsoid\"  upper hemisphere of an ellipsoid. params: {sx,sy,sz}\n"
"  \"torus\"      a ring with a hole. params: {major,minor}\n"
"                  major = distance from the centre to the tube's centre\n"
"                  minor = the tube's own radius (major MUST exceed minor)\n"
"  \"gaussian\"   localized bump on a flat plane. params: {amp,sigma}\n"
"  \"heightfield\" sinusoidal wave field. params: {amp,freq_x,freq_y,phase}\n"
"\n"
"=== BONES: THE RIGID SKELETON ===\n"
"A node may ALSO be a BONE -- a rigid segment with a pivot. That is what\n"
"makes a MECHANISM rather than a lump: a bone says which part rotates\n"
"relative to which, how long it is, and how much mass it carries.\n"
"\n"
"  \"bone\": {\n"
"    \"name\":      \"shaft\",           // optional, but needed to be a parent\n"
"    \"head\":      [x,y,z],            // REQUIRED: the pivot point\n"
"    \"tail\":      [x,y,z],            // REQUIRED: the far end of the segment\n"
"    \"bend\":      [x,y,z],            // optional: where the segment deviates\n"
"    \"radius\":    0.1,                // optional: cross-section radius\n"
"    \"mass\":      0.5,                // optional: mass contribution\n"
"    \"stiffness\": 1.0,                // optional: 0 = free to swing, 1 = rigid\n"
"    \"joints\":    [\"knee\"],          // optional: named articulation points\n"
"    \"material\":  \"steel\",           // optional: what this part is made of\n"
"    \"parent\":    \"housing\"          // optional: the PARENT BONE'S NAME\n"
"  }\n"
"\n"
"Rules for bones:\n"
"- A bone is given in the node's LOCAL frame, like every other coordinate.\n"
"- \"parent\" is a NAME, not an index. It must name another bone in the tree;\n"
"  an unknown name is REJECTED, and two bones may not share a name.\n"
"- A bone with a substance is a rigid part WITH skin (a limb, a shaft).\n"
"- A bone WITHOUT a substance is a bare axis or joint -- it contributes\n"
"  structure but no surface.\n"
"- Name the articulations where two bones meet, so a document can refer to\n"
"  \"the knee\" instead of an index.\n"
"\n"
"=== MATERIALS: what each part is made of ===\n"
"A bone's \"material\" names what that part is MADE OF. It is how a\n"
"technical document's materials become physics: the name is resolved\n"
"against a registry of REAL material properties, so a \"steel\" housing\n"
"carries steel's actual permeability, conductivity and heat capacity.\n"
"\n"
"Use ONLY these names (anything else is REJECTED):\n"
"  metals:    \"steel\", \"copper\", \"aluminium\", \"mumetal\",\n"
"             \"neodymium\" (a magnet)\n"
"  polymers:  \"ABS\", \"plastic\"\n"
"  ceramics:  \"glass\"\n"
"  fluids:    \"water\"\n"
"  biological:\"muscle\", \"bone\"\n"
"\n"
"Rules for materials:\n"
"- Put the material on the BONE, not on the node -- a bone is a rigid PART,\n"
"  and a part is made of one material.\n"
"- Omit it if the document does not say. An omitted material means \"not\n"
"  stated\", which is honest. Do NOT guess.\n"
"- A part with a substance AND a bone AND a material is a complete part:\n"
"  its shape, its articulation, and what it is made of.\n"
"\n"
"Rules:\n"
"- Use 1..12 nodes. Nest with \"children\".\n"
"- Overlapping systems BLEND; they never replace each other.\n"
"- Every node needs a substance OR children. At least one substance overall.\n"
"- Coordinates are local to the parent; the canvas spans -0.5..0.5.\n"
"- For a MECHANISM (a machine, a limb, anything with moving parts) give each\n"
"  moving part a bone. For a simple object, omit bones entirely.\n"
"- Reply with the JSON object only. No prose, no markdown fence.\n"
"\n"
"=== COARSENESS: PICK A LEVEL, NOT A PIXEL COUNT ===\n"
"Set \"lod\" on the ROOT to say how detailed this object needs to be. The\n"
"level decides both the surface resolution and how many bones are kept, so\n"
"you never have to think in pixels:\n"
"  0 = fine    near-final surface, every joint kept\n"
"  1 = medium  the default: a clean surface, every joint kept\n"
"  2 = coarse  a silhouette and its mass; small joints dropped\n"
"  3 = sketch  proportions and the main skeleton only\n"
"Choose 0 or 1 for a final asset, 2 for a background object, 3 for a\n"
"placeholder. Omit \"lod\" to get the default (1).\n"
"\n"
"Example:\n"
"{\"kind\":\"body\",\"color\":[0.6,0.7,0.8],\"children\":[\n"
"  {\"kind\":\"core\",\"substance\":\"ellipsoid\",\"substance_params\":{\"sx\":0.4,\"sy\":0.4,\"sz\":0.5}},\n"
"  {\"kind\":\"nodule\",\"origin\":[0.2,0.1,0.0],\"weight\":0.6,\n"
"   \"substance\":\"gaussian\",\"substance_params\":{\"amp\":0.12,\"sigma\":0.2},\n"
"   \"color\":[0.9,0.7,0.4]}]}\n"
"\n"
"Example (a ring):\n"
"{\"kind\":\"body\",\"substance\":\"torus\",\n"
" \"substance_params\":{\"major\":0.32,\"minor\":0.12},\"color\":[0.7,0.6,0.5]}\";\n"
"}\n"
"\n"
"Example (a mechanism: a shaft inside a housing, with a rotor):\n"
"{\"kind\":\"assembly\",\"children\":[\n"
"  {\"kind\":\"housing\",\"substance\":\"ellipsoid\",\n"
"   \"substance_params\":{\"sx\":0.4,\"sy\":0.4,\"sz\":0.4}},\n"
"  {\"kind\":\"shaft\",\"bone\":{\"name\":\"shaft\",\"head\":[0,0,-0.3],\"tail\":[0,0,0.3],\n"
"    \"radius\":0.05,\"mass\":0.4,\"joints\":[\"front\",\"rear\"]},\n"
"   \"substance\":\"plane\",\"substance_params\":{}},\n"
"  {\"kind\":\"rotor\",\"bone\":{\"name\":\"rotor\",\"parent\":\"shaft\",\n"
"    \"head\":[0,0,0],\"tail\":[0.2,0,0],\"radius\":0.12,\"mass\":0.2},\n"
"   \"substance\":\"ellipsoid\",\"substance_params\":{\"sx\":0.12,\"sy\":0.12,\"sz\":0.05}}]}\n";
}

std::string DescribeTree(const Element& root) {
    int nodes = 0, leaves = 0;
    CountTree(root, nodes, leaves);
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%d system(s), %d substance(s), kind=\"%s\"",
                  nodes, leaves, root.kind.c_str());
    return buf;
}

} // namespace AI
} // namespace DeliRium
