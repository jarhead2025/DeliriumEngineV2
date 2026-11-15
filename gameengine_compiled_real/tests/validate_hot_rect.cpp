// validate_hot_rect -- the UI-DECIDES-FIRST gate.
//
// THE BUG IT PINS: the workspace's MOUSEBUTTONDOWN handler ran BEFORE the
// UI pass, from raw geometry. The panels touch the viewport's edges (the
// tool strip, the render-path row, the console's generate button), so a
// click aimed at a button could land 1px inside the viewport rect and
// BOTH the tool and the button fired -- or an orbit started that ate the
// button's click. The user reported exactly this: "the buttons
// overlapping with the 3d screen dragging mechanisms".
//
// THE FIX: every interactive widget registers its rect (UI::Hot) during
// the pass; HotTest() reports whether a point is inside any of the LAST
// frame's rects; the workspace's viewport branch requires !HotTest().
// The map is one frame stale by design -- the same discipline ImGui's
// IsItemHovered uses.
//
// This gate asserts the CONTRACT of the hot-rect map on the real UI
// Context (no SDL window needed): registration, the one-frame delay,
// clearing, and that a widget that bails early is not hot.
#include "delirium/ui/UI.h"
#include <cstdio>
#include <string>

using namespace DeliRium::UI;

static int fails = 0;
static void check(const char* name, bool ok) {
    std::printf("%-64s %s\n", name, ok ? "OK" : "FAIL");
    if (!ok) ++fails;
}

int main() {
    Context c;
    c.mouse_x = 10; c.mouse_y = 10;

    // 1) Before any pass, the map is empty: a click anywhere reaches the
    //    viewport. (Frame 0 -- the app has just started.)
    check("an empty map vetoes nothing (frame 0)", !c.HotTest(10, 10) && c.HotEmpty());

    // 2) A widget registers its rect; the map updates on the NEXT
    //    BeginFrame, not immediately (the one-frame delay).
    c.Button(Rect{5, 5, 30, 12}, "t");
    check("a rect registered this pass is NOT yet hot", !c.HotTest(10, 10));
    c.BeginFrame(nullptr, nullptr, 100, 100);
    check("after the swap the widget IS hot", c.HotTest(10, 10));
    check("a point outside every rect is not hot", !c.HotTest(50, 50));

    // 3) A pass that registers nothing clears the map (a widget that
    //    disappeared no longer vetoes).
    c.BeginFrame(nullptr, nullptr, 100, 100);
    check("the map clears when a pass registers nothing", c.HotEmpty() && !c.HotTest(10, 10));

    // 4) Every widget family registers: button, toggle, icon, text field,
    //    number field, slider, check box, combo (with its open popup),
    //    tab bar, scroll area.
    {
        float v = 0.5f; bool b = false; std::string t = "x";
        int sel = 0, cur = 0, scroll = 0;
        const std::vector<std::string> items = {"a", "b"};
        const std::vector<std::string> tabs = {"one", "two"};
        c.Button(Rect{2, 2, 20, 10}, "b");
        c.ToggleButton(Rect{26, 2, 20, 10}, "tg", &b);
        c.IconButton(Rect{50, 2, 10, 10}, '+');
        c.TextField(Rect{2, 16, 40, 12}, "f", &t);
        c.NumberField(Rect{2, 32, 40, 12}, "n", &v, 0.1f, 0.f, 1.f);
        c.Slider(Rect{2, 48, 40, 10}, "s", &v, 0.f, 1.f);
        c.CheckBox(Rect{2, 62, 60, 12}, "cb", &b, "check");
        c.ComboBox(Rect{2, 78, 40, 12}, "combo", items, &sel);
        c.TabBar(Rect{2, 100, 80, 12}, "tabs", tabs, cur);
        c.ScrollArea(Rect{2, 116, 40, 30}, "scroll", &scroll, 100);
        c.BeginFrame(nullptr, nullptr, 200, 200);
        check("button hot",        c.HotTest(10, 5));
        check("toggle hot",        c.HotTest(30, 5));
        check("icon hot",          c.HotTest(54, 5));
        check("text field hot",    c.HotTest(10, 20));
        check("number field hot",  c.HotTest(10, 36));
        check("slider hot",        c.HotTest(10, 52));
        check("check box hot",     c.HotTest(10, 66));
        check("combo hot",         c.HotTest(10, 82));
        check("tab bar hot",       c.HotTest(10, 104));
        check("scroll area hot",   c.HotTest(10, 120));
    }

    // 5) A widget whose guard bails early renders and consumes nothing,
    //    so it must NOT be hot.
    {
        c.Slider(Rect{60, 20, 40, 10}, "n", nullptr, 0.f, 1.f);
        c.BeginFrame(nullptr, nullptr, 200, 200);
        check("a guarded-out widget is not hot", !c.HotTest(65, 24));
    }

    // 6) THE OVERLAP CONTRACT: a widget rect that pokes 1px into the
    //    viewport rect vetoes the click at that pixel. This is the exact
    //    geometry the user's bug lived in.
    {
        c.Button(Rect{100, 100, 40, 12}, "edge");   // bottom edge at y=112
        c.BeginFrame(nullptr, nullptr, 200, 200);
        const Rect viewport{80, 80, 200, 30};        // spans y 80..110
        check("the widget overlaps the viewport rect",
              viewport.Contains(110, 105) && c.HotTest(110, 105));
        check("a click in the overlap zone would be vetoed", c.HotTest(110, 105));
        check("a click deeper in the viewport is not vetoed",
              !c.HotTest(150, 100));
    }

    std::printf(fails ? "\nFAILURES: %d\n" : "\nALL CHECKS PASSED\n", fails);
    return fails ? 1 : 0;
}
