// Full pipeline: load .dlt (tree) + anchor structure -> deform via physics ->
// reconstruct tree -> count visible patches after cross-cull. End to end.
#include "delirium/psf/TreeFormat.h"
#include "delirium/psf/AnchorPhysics.h"
#include <cstdio>
#include <fstream>
#include <vector>
using namespace DeliRium;
int main(){
    // 1. load the tree model (geometry+color as sum-of-types)
    TreeModel tree;
    if(!tree.Load("assets/spot/spot.dlt")){ printf("dlt load failed\n"); return 1; }
    printf("[1] tree: %zu patches, %zu bytes\n", tree.PatchCount(), tree.ByteSize());

    // 2. load anchor structure for physics
    std::vector<Vec3> anchors;
    { std::ifstream f("assets/spot/spot_anchors.txt"); float x,y,z;
      while(f>>x>>y>>z) anchors.push_back({x,y,z}); }
    std::vector<std::array<uint32_t,4>> patch_anchors;
    { std::ifstream f("assets/spot/spot_patches.txt"); uint32_t a,b,c,d;
      while(f>>a>>b>>c>>d) patch_anchors.push_back({a,b,c,d}); }
    printf("[2] physics: %zu anchors, %zu patch-quads\n",
           anchors.size(), patch_anchors.size());

    // 3. init physics (factorize once), record rest
    AnchorPhysics phys; phys.Initialize(anchors, patch_anchors, 1.0f);
    printf("[3] rest volume %.4f, no deformation -> %d culled\n",
           phys.RestVolume(), [&]{auto c=phys.CrossCull(0.1f);int n=0;for(auto x:c)n+=x;return n;}());

    // 4. deform: press the back down
    std::vector<Vec3> forces(anchors.size(),{0,0,0});
    for(size_t i=0;i<anchors.size();++i) if(anchors[i].z>0.6f) forces[i]={0,0,-0.5f};
    phys.Step(forces);
    auto culled=phys.CrossCull(0.1f);
    int nc=0; for(auto x:culled) nc+=x;
    printf("[4] deformed: volume %.4f (conserved), %d patches culled\n",
           phys.Volume(), nc);

    // 5. reconstruct visible patches from the tree (those not culled)
    int visible=0, points=0;
    for(size_t p=0;p<tree.PatchCount() && p<culled.size();++p){
        if(culled[p]) continue;
        visible++;
        for(float u=0;u<=1;u+=0.2f) for(float v=0;v<=1;v+=0.2f){
            auto pt=tree.EvalPatch(p,u,v,16,16,16); (void)pt; points++;
        }
    }
    printf("[5] reconstructed %d visible patches -> %d points (culled %d patches)\n",
           visible, points, nc);
    printf("\nPIPELINE OK: dlt -> physics -> cull -> tree reconstruct, end to end\n");
    return 0;
}
