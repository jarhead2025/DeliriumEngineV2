// FILEPATH: engine/src/ai/ElementTreeParser.cpp
// JSON element tree -> Element. The AI -> geometry seam.
// Self-contained JSON reader (no nlohmann dependency -- that package is
// OPTIONAL in CMake) + a strict validator against the baker's vocabulary.
#include "delirium/ai/ElementTreeParser.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>

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
        const float minor = float(Param(params, "minor", Param(params, "tube_radius", 0.12)));
        const float major = float(Param(params, "major", Param(params, "ring_radius", 0.32)));
        if (minor <= 0.f) { why = "torus minor radius must be > 0"; return nullptr; }
        if (major <= minor) {
            why = "torus major radius must be greater than the minor radius";
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
    why = "unknown substance '" + kind +
          "' (use plane, ellipsoid/sphere, torus/donut, gaussian/bump, or heightfield/wave)";
    return nullptr;
}

// Build one element (and its subtree) from a JSON object.
bool BuildElement(const JsonValue& v, std::unique_ptr<Element>& out, std::string& why,
                  int depth);

// Build the children array of `v`.
bool BuildChildren(const JsonValue& v, Element& parent, std::string& why, int depth) {
    const JsonValue* kids = v.Find("children");
    if (!kids) return true;
    if (!kids->IsArray()) { why = "\"children\" must be an array"; return false; }
    for (const JsonValue& k : kids->items) {
        std::unique_ptr<Element> child;
        if (!BuildElement(k, child, why, depth + 1)) return false;
        parent.AddChild(std::move(child));
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
"Rules:\n"
"- Use 1..12 nodes. Nest with \"children\".\n"
"- Overlapping systems BLEND; they never replace each other.\n"
"- Every node needs a substance OR children. At least one substance overall.\n"
"- Coordinates are local to the parent; the canvas spans -0.5..0.5.\n"
"- Reply with the JSON object only. No prose, no markdown fence.\n"
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
" \"substance_params\":{\"major\":0.32,\"minor\":0.12},\"color\":[0.7,0.6,0.5]}";
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
