// validate_input_priority -- the WORKSPACE-level input contract.
//
// THE USER'S BUG: "the buttons overlapping with the 3d screen dragging
// mechanisms". The workspace's MOUSEBUTTONDOWN handler ran BEFORE the UI
// pass, from raw geometry, so a click aimed at a button that sat 1px
// inside the viewport rect did BOTH: it applied a sculpt tool AND fired
// the button (or started an orbit that ate the button's click).
//
// validate_hot_rect.cpp pins the UI-side contract (the hot-rect map).
// THIS gate pins the WORKSPACE-side contract: the decision that a click
// reaches the viewport. It is the function the input handler calls, and
// it is what the user's bug actually lived in.
//
// The decision has three clauses:
//   1. the GIZMO wins -- it is a viewport ACTION, not a UI widget;
//   2. a point outside the viewport rect is not a viewport action;
//   3. a point inside a UI WIDGET's rect is consumed by the UI, even
//      when the widget pokes into the viewport rect (the panels touch
//      the viewport's edges).
//
// NOTE: the widget rects come from the PREVIOUS frame (immediate mode
// is one frame stale by design -- the same discipline ImGui's
// IsItemHovered uses), so a widget that only just appeared cannot veto
// this frame's click. That is why the map is double-buffered.
#include "delirium/core/Workspace.h"
#include <cstdio>

using namespace DeliRium;

static int fails = 0;
static void check(const char* name, bool ok) {
    std::printf("%-68s %s\n", name, ok ? "OK" : "FAIL");
    if (!ok) ++fails;
}

int main() {
    // The Workspace only needs the Engine to hold a reference -- the
    // input decision and the layout are pure functions of the window
    // size and the UI's widget rects, so no window and no GPU are
    // needed here. (Engine is a singleton, so Instance() is how the
    // workspace gets its reference.)
    Workspace ws(Engine::Instance());
    ws.ComputeLayout(1000, 700);
    const auto& L = ws.GetLayout();

    // --- the layout must be sane, or every assertion below is vacuous ---
    check("the viewport is non-empty", L.viewport.w > 0 && L.viewport.h > 0);
    check("the viewport sits between the panels",
          L.viewport.x >= L.left.x + L.left.w &&
          L.viewport.x + L.viewport.w <= L.right.x);
    check("the console is below the viewport",
          L.console.y >= L.viewport.y + L.viewport.h);

    // The viewport's CENTRE is deep inside the viewport and inside no
    // panel -- the baseline case.
    const int cx = L.viewport.x + L.viewport.w / 2;
    const int cy = L.viewport.y + L.viewport.h / 2;
    check("the viewport centre reaches the viewport",
          ws.ClickReachesViewport(cx, cy));

    // --- clause 2: outside the viewport is never a viewport action ---
    check("a point in the left panel does not reach the viewport",
          !ws.ClickReachesViewport(L.left.x + 2, L.left.y + 2));
    check("a point in the right panel does not reach the viewport",
          !ws.ClickReachesViewport(L.right.x + 2, L.right.y + 2));
    check("a point in the console does not reach the viewport",
          !ws.ClickReachesViewport(L.console.x + 2, L.console.y + 2));
    check("a point in the status bar does not reach the viewport",
          !ws.ClickReachesViewport(L.status.x + 2, L.status.y + 2));

    // --- clause 3: a WIDGET that pokes into the viewport vetoes it ---
    //
    // This is the exact geometry the bug lived in. The widget is
    // registered in the UI pass, and the map is one frame stale -- so the
    // veto applies from the NEXT frame on. The workspace owns the UI
    // context, so the gate drives it the way the app does: draw, then
    // ask.
    //
    // The widget is placed so it straddles the viewport's RIGHT edge --
    // exactly where the tool strip and the inspector's buttons live.
    const UI::Rect edge_button{ L.viewport.x + L.viewport.w - 4,
                                L.viewport.y + 10, 40, 12 };
    const int probe_x = edge_button.x + 2;   // 2px INSIDE the viewport
    const int probe_y = edge_button.y + 6;
    check("the probe point is inside the viewport rect",
          L.viewport.Contains(probe_x, probe_y));
    check("the probe point is inside the widget rect",
          edge_button.Contains(probe_x, probe_y));

    // Before the widget registers, the viewport still gets the click.
    check("before the widget registers, the viewport gets the click",
          ws.ClickReachesViewport(probe_x, probe_y));

    // Register it through the real UI context and let a frame pass.
    ws.GetUI().Button(edge_button, "edge");
    ws.GetUI().BeginFrame(nullptr, nullptr, L.win_w, L.win_h);
    check("AFTER the widget registers, the viewport is VETOED",
          !ws.ClickReachesViewport(probe_x, probe_y));
    check("the veto does not spread: deeper in the viewport still works",
          ws.ClickReachesViewport(cx, cy));
    check("the veto does not spread: the panel is still not the viewport",
          !ws.ClickReachesViewport(L.left.x + 2, L.left.y + 2));

    // --- clause 1: the GIZMO wins over everything ---
    // The gizmo lives INSIDE the viewport's top-right corner. A click on
    // it must be a viewport action (it rotates the camera), and it must
    // survive even if a widget overlaps it.
    //
    // THE CHECK MUST DISCRIMINATE: the gizmo's centre is inside the
    // viewport, so the plain viewport clause would also return true --
    // a gate that cannot distinguish the gizmo clause from the viewport
    // clause is not testing the gizmo. So the widget is registered OVER
    // the gizmo: with the gizmo clause the click is still a viewport
    // action, and without it the UI vetoes it.
    const UI::Rect g = ws.GizmoRect();
    check("the gizmo rect is non-empty", g.w > 0 && g.h > 0);
    if (g.w > 0 && g.h > 0) {
        const int gx = g.x + g.w / 2;
        const int gy = g.y + g.h / 2;
        check("the gizmo centre is inside the viewport",
              L.viewport.Contains(gx, gy));

        // A widget straddling the gizmo would veto the gizmo if the
        // gizmo did not win.
        const UI::Rect over_gizmo{ g.x - 2, g.y - 2, g.w + 4, g.h + 4 };
        ws.GetUI().Button(over_gizmo, "over-gizmo");
        ws.GetUI().BeginFrame(nullptr, nullptr, L.win_w, L.win_h);

        // Without the gizmo clause, the widget's veto would apply.
        // With it, the gizmo wins. THIS is the discriminating test.
        check("the gizmo WINS over a widget that overlaps it",
              ws.ClickReachesViewport(gx, gy));
    }

    std::printf(fails ? "\nFAILURES: %d\n" : "\nALL CHECKS PASSED\n", fails);
    return fails ? 1 : 0;
}
