//
//  ShaderDefinitions.h
//
//  Created by brien dieterle on 5/24/20.
//  Copyright © 2020 brien dieterle. All rights reserved.
//

#ifndef ShaderDefinitions_h
#define ShaderDefinitions_h

#endif /* ShaderDefinitions_h */
#include <simd/simd.h>

struct Vertex {
    vector_float2 pos;
    vector_float2 textureCoordinate;
};

struct ColorSample {
    vector_float4 color[4];
};

struct SpectralColorArray {
    vector_float4 color[3];
};

struct Dab {
    // color "reflectance" of pigment
    // 12 channels, log2 encoded
    vector_float4 color[3];
    // center of dab in global coordinates
    vector_float2 pos;
    // radius of dab in pixels
    float radius;
    // thickness of dab (0-1)
    float volume;
    // opaqueness of dab (0-1
    float opacity;
    // strength of dab to replace the existing data (0-1)
    float strength;
    // feathering of dab edges (0-1)
    float hardness;
    // whether to erase or not (can be partial) (0-1)
    float eraser;
    // weird thing to skip draw if smudge thickness is
    // out of range (-1 to +1)
    float smudgeThicknessThreshold;
    // ratio of smudge color to brush color to use for painting
    float smudgeAmount;
    // which smudge texture index to use for this dab
    uint16_t smudgeBucket;
    // ratio of canvas sample to smudge bucket sample
    // 0.0 immediately replaces smudge bucket, 1.0 never does
    float smudgeLength;
    // the pressure from the input device
    float pressure;
    // the sample radius to sample canvas for smudge bucket updates
    float smudgeRadius;
    // the aspect ratio of the dab.  (0.1-1.0)
    float dabRatio;
    // rotation of dab
    float dabAngle;
    // ellipse vs squircle
    float dabShape;
    // modifying factor for squircle, can vary shape a lot
    float dabShapeMod;
    // odds of jittering smudge
    float smudgeJitterChance;
    // how much to jitter smudge if we do
    float smudgeJitter;
    // how much to jitter value/brightness of brush color; darker-lighter (-1.0-1.0
    float valueJitter;
    // how much to jitter shift color left or right (on spectrum) (-1.0-1.0)
    float colorJitter;
    // odds of shifting value/brightness
    float valueJitterChance;
    // odds of shifting color
    float colorJitterChance;
    // how much to jitter volume/thickness (0-1)
    float volumeJitter;
    // odds of jittering volume/thickness
    float volumeJitterChance;
    // how much to jitter opacity
    float opacityJitter;
    // odds of jittering opacity
    float opacityJitterChance;
    // odds of drawing a pixel at all
    float dabCoverage;
    // how stong to pick up color from layer below active layer
    // only when smudging
    float solvent;
};


// information that is applicable for all the dabs
// in the dabarray
struct DabMeta {
    // how many dabs to draw in this draw call
    // should loop over dabArray
    uint32_t dabCount;
    // global origin coordinates of texture
    vector_float2 texOrigin;
    // whether we are only updating the smudge
    // not used for brush drawing at all
    uint8_t updateSmudgeOnly;
    // whether the draw call includes the lower layer
    // below the active layer as an additional texture
    uint8_t hasLowerTexture;
    // some random seeds, six total
    // useful in noise generators
    vector_int4 randomSeeds;
    vector_int2 randomSeeds2;
};

struct OverOp {
    float srcOpacity;
    float dstOpacity;
    float srcThickness;
    float dstThickness;
//    uint8_t layerMode;
};
