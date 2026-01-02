//
//  Shaders.metal
//
//  Created by brien dieterle on 5/24/20.
//  Copyright © 2020 brien dieterle. All rights reserved.
//

#include <metal_stdlib>
#include <metal_relational>
#include "ShaderDefinitions.h"
using namespace metal;


constant half3 displayP3Luma = half3(0.265667693, 0.691738522, 0.0451133819);
constant half EPSILON = 0.0002; // for some reason 0.0001 is too small, NaNs
constant half offset = 1.0 - EPSILON;
constant half EPSILON_LOG = -12.287712379549449;




struct VertexOut {
    float4 color;
    float4 pos [[position]];
};

typedef struct
{
    float4 clipSpacePosition [[position]];
    float2 textureCoordinate;

} RasterizerData;


kernel void rgbToSpectral(texture2d <half, access::read> rgbTexture [[texture(0)]],
                          texture2d_array <half, access::read_write> spectralTexture [[texture(1)]],
                          texture2d <half, access::read> greyTexture [[texture(2)]],
                          constant SpectralColorSpace *colorSpace [[ buffer(0) ]],
                          uint2 gid [[thread_position_in_grid]]) {
    half4 rgba = rgbTexture.read(gid);
    half4 greyRGBA = greyTexture.read(gid);
    half alpha = rgba.w;
    if (alpha > 0.0) {
        rgba /= alpha;
    }
    rgba = clamp(rgba, 0.0, 1.0);
    rgba = (rgba * offset) + EPSILON;
    
    // if greyscale, just use that value for all channels
    // otherwise use color
    
    float4 colorShort;
    float4 colorMedium;
    float4 colorLong;
    if ( rgba.x == rgba.y == rgba.z ) {
        colorShort = float4(rgba.x, rgba.x, rgba.x, rgba.x );
        colorMedium = float4(rgba.x, rgba.x, rgba.x, rgba.x );
        colorLong = float4(rgba.x, rgba.x, rgba.x, rgba.x );
    } else {
        colorShort = colorSpace->red[0] * rgba.x + colorSpace->green[0] * rgba.y + colorSpace->blue[0] * rgba.z;
        colorMedium = colorSpace->red[1] * rgba.x + colorSpace->green[1] * rgba.y + colorSpace->blue[1] * rgba.z;
        colorLong = colorSpace->red[2] * rgba.x + colorSpace->green[2] * rgba.y + colorSpace->blue[2] * rgba.z;
    }
    
    spectralTexture.write(half4(log2(colorShort) * alpha), gid, 0);
    spectralTexture.write(half4(log2(colorMedium) * alpha), gid, 1);
    spectralTexture.write(half4(log2(colorLong) * alpha), gid, 2);
    spectralTexture.write(half4(alpha, max(greyRGBA.x * half(10.0), EPSILON), 1.0, 0.0), gid, 3);
    
}

kernel void rgbToGreyscale(texture2d <half, access::read_write> rgbTexture [[texture(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
    
    half4 rgba = rgbTexture.read(gid);
    
    half alpha = rgba.w;
    if (alpha > 0.0) {
        rgba /= alpha;
    }
    
   
    rgba = clamp(rgba, 0.0, 1.0);
    half grey = dot(rgba.rgb, displayP3Luma);

    
    rgbTexture.write(half4(grey, grey, grey, 1.0), gid);
}

// assume already linear spectral
kernel void spectralToRGB(texture2d_array <half, access::read> spectralTexture [[texture(0)]],
                          texture2d <half, access::read_write> rgbTexture [[texture(1)]],
                          constant SpectralColorSpace *colorSpace [[ buffer(0) ]],
                          uint2 gid [[thread_position_in_grid]]) {

    // read spectral array texture and write to an RGB texture with associated alpha
    
    half4 srcS = spectralTexture.read(gid, 0);
    half4 srcM = spectralTexture.read(gid, 1);
    half4 srcL = spectralTexture.read(gid, 2);
    
    half4 S = (srcS);
    half4 M = (srcM);
    half4 L = (srcL);
    half spd [12] = {S.x, S.y, S.z, S.w, M.x, M.y, M.z, M.w, L.x, L.y, L.z, L.w};
    
    half3 rgb = {0};
    
    // convert back to RGB
    for (int i=0; i<12; i++) {
        rgb[0] += colorSpace->T_MATRIX[0][i] * spd[i];
        rgb[1] += colorSpace->T_MATRIX[1][i] * spd[i];
        rgb[2] += colorSpace->T_MATRIX[2][i] * spd[i];
    }
    
    
    // undo offset
    half3 resColor = ((rgb - EPSILON) / offset);
    
    //resColor = saturate(resColor);
    
    // apply alpha to pixels so that white=black, invert for normal programs.  Only means anything with background disabled
    half4 renderedPixel = half4(resColor, 1.0);
    rgbTexture.write(renderedPixel, gid); //write to render target

    
    
}

kernel void spectralLogToRGB(texture2d_array <half, access::read> spectralTexture [[texture(0)]],
                          texture2d <half, access::read_write> rgbTexture [[texture(1)]],
                          constant SpectralColorSpace *colorSpace [[ buffer(0) ]],
                          uint2 gid [[thread_position_in_grid]]) {

    // read spectral array texture and write to an RGB texture with associated alpha
    
    half4 srcS = spectralTexture.read(gid, 0);
    half4 srcM = spectralTexture.read(gid, 1);
    half4 srcL = spectralTexture.read(gid, 2);
    
    // log to linear
    half4 S = exp2(srcS);
    half4 M = exp2(srcM);
    half4 L = exp2(srcL);
    half spd [12] = {S.x, S.y, S.z, S.w, M.x, M.y, M.z, M.w, L.x, L.y, L.z, L.w};
    
    half3 rgb = {0};
    
    // convert back to RGB
    for (int i=0; i<12; i++) {
        rgb[0] += colorSpace->T_MATRIX[0][i] * spd[i];
        rgb[1] += colorSpace->T_MATRIX[1][i] * spd[i];
        rgb[2] += colorSpace->T_MATRIX[2][i] * spd[i];
    }
    
    
    // undo offset
    
    half3 resColor = ((rgb - EPSILON) / offset);
    //resColor = saturate(resColor);
    
    half4 renderedPixel = half4(resColor, 1.0);
    rgbTexture.write(renderedPixel, gid); //write to render target
    
}




kernel void updateSmudgeBuckets(constant Dab *dabArray [[ buffer(0) ]],
                    constant DabMeta *dabMeta [[ buffer(1) ]],
                    texture2d_array <half, access::read> canvas [[texture(0)]],
                    texture2d_array <half, access::read_write> smudgeBuckets [[texture(1)]],
                    uint2 gid [[thread_position_in_grid]]) {


    int dabCount = dabMeta->dabCount;

    for (int dabIndex=0; dabIndex < dabCount; dabIndex++) {
        
        
        half smudgeLength = dabArray[dabIndex].smudgeLength;
        uint2 bucket = uint2(dabArray[dabIndex].smudgeBucket, 0);
        // extra row/area for more metadata
        uint2 bucketMeta = uint2(dabArray[dabIndex].smudgeBucket, 1);
        half4 smudgeBucketMeta = smudgeBuckets.read(bucketMeta, 0);

        // how recent this bucket was updated, break early
        half recentness = smudgeBucketMeta.x;
        // smudgeBucketMeta.y signals that we are initted and sampled
        
        // reset recentness when we sample
        if (recentness < 0.5) {
            if (recentness == 0.0) {
                smudgeLength = 0.0;
            }
            recentness = 1.0;
            
        } else {
            smudgeBuckets.write(half4(recentness * smudgeLength, smudgeBucketMeta.y, 1, 1), bucketMeta, 0);
            smudgeBuckets.fence();
            continue;
        }
        

        
       
        if (smudgeLength >= 1.0 && smudgeBucketMeta.y == 1) continue;
        
        
       
        
        half2 center = half2(dabArray[dabIndex].pos);
        half dist = distance(half2(gid) + half2(dabMeta->texOrigin), center);
        if (dist > dabArray[dabIndex].smudgeRadius + 2) continue; // skip sampling beyond the smudge radius
        
        smudgeBuckets.write(half4(recentness, 1, 1, 1), bucketMeta, 0);
        
        half4 smudgeSampleD = 0;
        smudgeSampleD = canvas.read(gid, 3);
        
        // Only smudge when canvas thickness is above or below a threshhold
        half smudgeThicknessThreshold = dabArray[dabIndex].smudgeThicknessThreshold;
        if (smudgeThicknessThreshold > 0.0 && smudgeSampleD.y / 10.0 < smudgeThicknessThreshold) {
            //continue;
            smudgeLength += (smudgeThicknessThreshold - smudgeSampleD.y / 10.0) ;
        } else if (smudgeThicknessThreshold < 0.0 && smudgeSampleD.y / 10.0 < -smudgeThicknessThreshold) {
            //continue;
            smudgeLength += ( -smudgeThicknessThreshold - smudgeSampleD.y / 10.0) ;
            
        }
        
        
        // bucket to update/average into

        
        half4 smudgeBucketA = smudgeBuckets.read(bucket, 0);
        half4 smudgeBucketB = smudgeBuckets.read(bucket, 1);
        half4 smudgeBucketC = smudgeBuckets.read(bucket, 2);
        half4 smudgeBucketD = smudgeBuckets.read(bucket, 3);

        
        // sample the canvas
        half4 smudgeSampleA = 0;
        half4 smudgeSampleB = 0;
        half4 smudgeSampleC = 0;
       


        smudgeSampleA = canvas.read(gid, 0);
        smudgeSampleB = canvas.read(gid, 1);
        smudgeSampleC = canvas.read(gid, 2);
      
        
//       // reweight smudge by volumes
//
//        half volTotal = smudgeSampleD.y + smudgeBucketD.y;
//        
//        if (volTotal > 0.0) {
//            smudgeLength *= (smudgeBucketD.y / volTotal);
//        }
        
        smudgeLength = clamp(smudgeLength, half(0.0), half(1.0));

        
        
        smudgeBucketA = smudgeBucketA * smudgeLength + (1.0 - smudgeLength) * smudgeSampleA;
        smudgeBucketB = smudgeBucketB * smudgeLength + (1.0 - smudgeLength) * smudgeSampleB;
        smudgeBucketC = smudgeBucketC * smudgeLength + (1.0 - smudgeLength) * smudgeSampleC;
        smudgeBucketD = smudgeBucketD * smudgeLength + (1.0 - smudgeLength) * smudgeSampleD;
         
        

        if (any(isnan(smudgeBucketA)) || any(isnan(smudgeBucketD)) || any(isinf(smudgeBucketA)) || any(isinf(smudgeBucketD))) {
            return;
        }
        
        smudgeBuckets.write(smudgeBucketA, bucket, 0);
        smudgeBuckets.write(smudgeBucketB, bucket, 1);
        smudgeBuckets.write(smudgeBucketC, bucket, 2);
        smudgeBuckets.write(smudgeBucketD, bucket, 3);


        smudgeBuckets.fence();
        
    }

}



kernel void applyBumpMap(texture2d_array <half, access::read_write> canvas [[texture(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    
    half4 cS = exp2(canvas.read(gid, 0));
    half4 cM = exp2(canvas.read(gid, 1));
    half4 cL = exp2(canvas.read(gid, 2));
    half4 cMeta = canvas.read(gid,3);
        
    half Gx = 0;
    half Gy = 0;

    // East
    Gx -= canvas.read(gid - uint2(1, 0), 3).y * 20;
    Gx -= canvas.read(gid - uint2(2, 0), 3).y * 10;

    // West
    Gx += canvas.read(gid + uint2(1, 0), 3).y * 20;
    Gx += canvas.read(gid + uint2(2, 0), 3).y * 10;

    
    // North-East
    Gx -= canvas.read(gid - uint2(-1, 1), 3).y * 10;
    Gx -= canvas.read(gid - uint2(-2, 2), 3).y * 5;
    Gx -= canvas.read(gid - uint2(-2, 1), 3).y * 8;
    Gx -= canvas.read(gid - uint2(-1, 2), 3).y * 4;
    
    // North-West
    Gx += canvas.read(gid - uint2(1, 1), 3).y * 10;
    Gx += canvas.read(gid - uint2(2, 2), 3).y * 5;
    Gx += canvas.read(gid - uint2(2, 1), 3).y * 8;
    Gx += canvas.read(gid - uint2(1, 2), 3).y * 4;
    
    // South-East
    Gx -= canvas.read(gid + uint2(1, 1), 3).y * 10;
    Gx -= canvas.read(gid + uint2(2, 2), 3).y * 5;
    Gx -= canvas.read(gid + uint2(2, 1), 3).y * 8;
    Gx -= canvas.read(gid + uint2(1, 2), 3).y * 4;
    
    // South-West
    Gx += canvas.read(gid + uint2(-1, 1), 3).y * 10;
    Gx += canvas.read(gid + uint2(-2, 2), 3).y * 5;
    Gx += canvas.read(gid + uint2(-2, 1), 3).y * 8;
    Gx += canvas.read(gid + uint2(-1, 2), 3).y * 4;
    

    // North
    Gy -= canvas.read(gid - uint2(0, 1), 3).y * 20;
    Gy -= canvas.read(gid - uint2(0, 2), 3).y * 10;

    // South
    Gy += canvas.read(gid + uint2(0, 1), 3).y * 20;
    Gy += canvas.read(gid + uint2(0, 2), 3).y * 10;
    
    // North-East
    Gy -= canvas.read(gid - uint2(-1, 1), 3).y * 10;
    Gy -= canvas.read(gid - uint2(-2, 2), 3).y * 5;
    Gy -= canvas.read(gid - uint2(-1, 2), 3).y * 8;
    Gy -= canvas.read(gid - uint2(-2, 1), 3).y * 4;

    // North-West
    Gy -= canvas.read(gid - uint2(1, 1), 3).y * 10;
    Gy -= canvas.read(gid - uint2(2, 2), 3).y * 5;
    Gy -= canvas.read(gid - uint2(1, 2), 3).y * 8;
    Gy -= canvas.read(gid - uint2(2, 1), 3).y * 4;

    // South-East
    Gy += canvas.read(gid + uint2(1, 1), 3).y * 10;
    Gy += canvas.read(gid + uint2(2, 2), 3).y * 5;
    Gy += canvas.read(gid + uint2(1, 2), 3).y * 8;
    Gy += canvas.read(gid + uint2(2, 1), 3).y * 4;

    // South-West
    Gy += canvas.read(gid + uint2(-1, 1), 3).y * 10;
    Gy += canvas.read(gid + uint2(-2, 2), 3).y * 5;
    Gy += canvas.read(gid + uint2(-1, 2), 3).y * 8;
    Gy += canvas.read(gid + uint2(-2, 1), 3).y * 4;
    
    // cook-torrance adapted from https://github.com/pboechat/cook_torrance/blob/master/LICENSE
    float scale = 0.0025;
    float3 normal = 0;
    float3 lightDir = normalize(float3(0.3, 0.3, 1.0));
    float3 viewDir = float3(0,0,1.0);
    float F0 = 0.3;
    float roughness = clamp(float(cMeta.w), 0.1, 1.0);
    float k = 0.2;
    normal.x = scale * Gx;
    normal.y = scale * Gy;
    normal.z = 1.0;
    
    normal = normalize(normal);
    float NdotL = dot(normal, lightDir);
    
    float Rs = 0.0;
    if (NdotL > 0) {
        
        float3 H = normalize(lightDir + viewDir);
        float NdotH = max(0.0, dot(normal, H));
        float NdotV = max(0.0, dot(normal, viewDir));
        float VdotH = max(0.0, dot(lightDir, H));

        // Fresnel reflectance
        float F = pow(1.0 - VdotH, 5.0);
        F *= (1.0 - F0);
        F += F0;

        // Microfacet distribution by Beckmann
        float m_squared = roughness * roughness;
        float r1 = 1.0 / (4.0 * m_squared * pow(NdotH, 4.0));
        float r2 = (NdotH * NdotH - 1.0) / (m_squared * NdotH * NdotH);
        float D = r1 * exp(r2);

        // Geometric shadowing
        float two_NdotH = 2.0 * NdotH;
        float g1 = (two_NdotH * NdotV) / VdotH;
        float g2 = (two_NdotH * NdotL) / VdotH;
        float G = min(1.0, min(g1, g2));

        Rs = (F * D * G) / (M_PI_F * NdotL * NdotV);
        
        cS = cS * NdotL + NdotL * cMeta.x * clamp(float(cMeta.y), 0.0, 0.05) * (k + Rs * (1.0 - k));
        cM = cM * NdotL + NdotL * cMeta.x * clamp(float(cMeta.y), 0.0, 0.05) * (k + Rs * (1.0 - k));
        cL = cL * NdotL + NdotL * cMeta.x * clamp(float(cMeta.y), 0.0, 0.05) * (k + Rs * (1.0 - k));
        
    }
    
    
    
    canvas.write(log2(cS), gid, 0);
    canvas.write(log2(cM), gid, 1);
    canvas.write(log2(cL), gid, 2);
    
}



// reduce the paint of dst based on jaggies of src

kernel void reducePaint(texture2d_array <half, access::read> src [[texture(0)]],
                        texture2d_array <half, access::read_write> dst [[texture(1)]],
                        //constant dabMeta *dabMeta [[ buffer(0) ]],
                        //constant float2 &bumpOpts [[ buffer(0) ]],
                        uint2 gid [[thread_position_in_grid]]) {
    
    half4 dstS = dst.read(gid, 0);
    half4 dstM = dst.read(gid, 1);
    half4 dstL = dst.read(gid, 2);
    half4 dstMeta = dst.read(gid, 3);
    //half volume = dstMeta.x > 0.0 ? dstMeta.y / dstMeta.x : dstMeta.y;
    //half depth =  clamp(src.read(gid, 3).y + half(dstMeta.y / 2.0) + half(dstMeta.w / 10.0), half(0.0), half(1.0));
    half depth =  clamp(src.read(gid, 3).y + half(dstMeta.y / 10.0), half(0.0), half(1.0));
    //if (depth != 0.0) {
        dstS *= depth;
        dstM *= depth;
        dstL *= depth;
        dstMeta.x *= depth;  // reduce alpha
        dstMeta.y *= depth; // reduce volume

   // }

    dst.write(dstS, gid, 0);
    dst.write(dstM, gid, 1);
    dst.write(dstL, gid, 2);
    dst.write(dstMeta, gid, 3);
   // }
}


// add fillColor to dst texture (in log, so like a multiply)
kernel void fillWithColor(constant ColorSample *fillColor [[ buffer(0) ]],
                        texture2d_array <half, access::read_write> dst [[texture(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
    
    
    half4 dstS = dst.read(gid, 0);
    half4 dstM = dst.read(gid, 1);
    half4 dstL = dst.read(gid, 2);
    //half4 dstMeta = dst.read(gid, 3);
    
    dst.write(half4(fillColor->color[0]) + dstS, gid, 0);
    dst.write(half4(fillColor->color[1]) + dstM, gid, 1);
    dst.write(half4(fillColor->color[2]) + dstL, gid, 2);
    
    
}



kernel void sampleCanvas(volatile device ColorSample *colorSample [[ buffer(0) ]],
                         texture2d_array <half, access::read> src [[texture(0)]]) {
    
    uint width = src.get_width();
    uint height = src.get_height();
    //uint pixels = width * height;
    
    float sumWeights = 0.0;
    for (uint i=0; i < width; i++) {
        for (uint j=0; j < height; j++) {
            uint2 pos = uint2(i, j);
            half4 a = src.read(pos, 3);
                sumWeights += a.y;
                    }
    }
    
    if (sumWeights > 0.0) {
        for (uint i=0; i < width; i++) {
            for (uint j=0; j < height; j++) {
                uint2 pos = uint2(i, j);
                
                half4 s = src.read(pos, 0);
                half4 m = src.read(pos, 1);
                half4 l = src.read(pos, 2);
                half4 a = src.read(pos, 3);
                
               
                half beerFac = a.z > 0.0 ? a.z : 1.0; //a.x > 0.0 ? a.x : 1.0;
              
                float myWeight = (a.y / sumWeights) ;
                colorSample->color[0] += vector_float4(s / beerFac) * myWeight;
                colorSample->color[1] += vector_float4(m / beerFac) * myWeight;
                colorSample->color[2] += vector_float4(l / beerFac) * myWeight;
                colorSample->color[3] += vector_float4(a) * myWeight;
            }
        }
    }

    
}


// Generate a random float in the range [0.0f, 1.0f] using x, y, and z (based on the xor128 algorithm)
float rand(int x, int y, int z)
{
    int seed = x + y * 57 + z * 241;
    seed= (seed<< 13) ^ seed;
    return (( 1.0 - ( (seed * (seed * seed * 15731 + 789221) + 1376312589) & 2147483647) / 1073741824.0f) + 1.0f) / 2.0f;
}

// draw a radial sweep between N munsell colors into a texture

kernel void drawRadialSweep(constant SpectralColorArray *spectralColorArray [[ buffer(0) ]],
                            constant vector_float2 &center [[ buffer(1) ]],
                            constant float &radius [[ buffer(2) ]],
                            constant float &slider [[ buffer(3) ]],
                            constant uint &numColors [[ buffer (4) ]],
                            constant float &satSlider [[ buffer(5) ]],
                            constant uint &logEncoding [[ buffer(6)]],
                            texture2d_array <half, access::read_write> wheelTexture [[ texture(0) ]],
                            uint2 gid [[thread_position_in_grid]]){
    
    float2 centerPos = float2(center);
    float sliderVal = max(float(EPSILON), smoothstep(0.0, 1.0, slider));
    float dist = clamp(distance(centerPos, float2(gid)) / float(radius * 0.90), float(0.0), float(1.0));
    
    half4 grey;
    
    if (logEncoding >= 1) {
        dist = pow(smoothstep(0.0, 1.0, dist), 2.0);
        
        grey = log2(half4(sliderVal, sliderVal, sliderVal, sliderVal)) * (1.0 - dist);
        
    } else {
        dist = pow(smoothstep(0.0, 1.0, dist), 2.0);
        grey = (half4(sliderVal, sliderVal, sliderVal, sliderVal)) * (1.0 - dist);
    }
    
    //int colorIndex = acos(dot(centerPos, half2(gid)));
    float colorIndex = 0;
    
    float angle = atan2(float(gid.y) - centerPos.y, float(gid.x) - centerPos.x) / (2.0 * M_PI_H);
    if (angle < 0.0) {
        angle += 1.0;
    }
    colorIndex = clamp(angle, 0.0, 1.0) * numColors;
    //half remainder;
    float rem;
////    if (logEncoding == 1) {
        rem = smoothstep(0.0, 1.0, fract(colorIndex));
//    } else {
//        rem = fract(colorIndex);
//    }
    uint colorIndexAdjacent = colorIndex + 1;

    
    if (colorIndexAdjacent >= numColors) {
        colorIndexAdjacent = 0;
    }
    
    half4 col1 = (half4(spectralColorArray[int(colorIndex)].color[0]) *
        (1.0 - rem)) + (rem * half4(spectralColorArray[colorIndexAdjacent].color[0]));
    
    half4 col2 = (half4(spectralColorArray[int(colorIndex)].color[1]) *
        (1.0 - rem)) + (rem * half4(spectralColorArray[colorIndexAdjacent].color[1]));
    
    half4 col3 = (half4(spectralColorArray[int(colorIndex)].color[2]) *
        (1.0 - rem)) + (rem * half4(spectralColorArray[colorIndexAdjacent].color[2]));
    
    if (logEncoding >= 1) {
        wheelTexture.write((col1 * satSlider) * dist + grey, gid, 0);
        wheelTexture.write((col2 * satSlider) * dist + grey, gid, 1);
        wheelTexture.write((col3 * satSlider) * dist + grey, gid, 2);
    } else {
        wheelTexture.write((col1 + pow((1.0 - satSlider), 2.5)) * dist + grey, gid, 0);
        wheelTexture.write((col2 + pow((1.0 - satSlider), 2.5)) * dist + grey, gid, 1);
        wheelTexture.write((col3 + pow((1.0 - satSlider), 2.5)) * dist + grey, gid, 2);
    }
    
    wheelTexture.write(half4(1,1,1,1), gid, 3);
    
}

kernel void drawSquareSweep(constant SpectralColorArray *spectralColorArray [[ buffer(0) ]],
                            constant vector_float2 &center [[ buffer(1) ]],
                            constant float &radius [[ buffer(2) ]],
                            constant float &slider [[ buffer(3) ]],
                            constant uint &numColors [[ buffer (4) ]],
                            constant float &satSlider [[ buffer(5) ]],
                            constant uint &logEncoding [[ buffer(6)]],
                            texture2d_array <half, access::read_write> wheelTexture [[ texture(0) ]],
                            uint2 gid [[thread_position_in_grid]]){
    
//    float2 centerPos = float2(center);
    //float sliderVal = max(float(EPSILON), smoothstep(0.0, 1.0, slider));
    //float dist = clamp(distance(centerPos, float2(gid)) / float(radius * 0.90), float(0.0), float(1.0));
    float distX = 1.0 - clamp((wheelTexture.get_width() - float(gid.x)) / wheelTexture.get_width(), float(0.0), float(1.0));
    float distY = 1.0 -clamp((wheelTexture.get_height() - float(gid.y)) / wheelTexture.get_height(), float(0.0), float(1.0));
    
    half4 blackness;
    
//    if (logEncoding == 1) {
//        //dist = pow(smoothstep(0.0, 1.0, dist), 2.0);
//        
//        grey = log2(half4(distX, distX, distX, distX));
//        
//    } else {
        //dist = pow(smoothstep(0.0, 1.0, dist), 2.0);
//    distY = pow(smoothstep(0.0, 1.0, distY), 2.0);
//    distX = pow(smoothstep(0.0, 1.0, distX), 2.0);

    
    
    blackness = (half4(EPSILON_LOG, EPSILON_LOG, EPSILON_LOG, EPSILON_LOG))  * pow(distY, 1.0) * pow(distX, 1.0);
//    half whiteness = clamp(((pow(distX, 1.0) + pow(distY, 1.0))) , 0.0, 1.0);
    half whiteness = clamp(((pow(distX, 1.0) + pow(distY, 1.0))) / 2.0 , 0.0, 1.0);

    half greyRatio = clamp((1.0 - distX), 0.0, 1.0);
    half grey = max(half(pow(slider, 2.0)), EPSILON);
    half4 greyness = log2((half4(grey, grey, grey, grey))) * greyRatio;

    
//    }
    
    float colorIndex = 0;
    
    
    colorIndex = clamp(satSlider == 1 ? 0.0 : satSlider, 0.0, 1.0) * (numColors);
    //half remainder;
//    float rem;
////    if (logEncoding == 1) {
    float rem = fract(colorIndex);
//    } else {
//        rem = fract(colorIndex);
//    }
    uint colorIndexAdjacent = uint(colorIndex) + 1;

    
    if (colorIndexAdjacent >= uint(numColors)) {
        colorIndexAdjacent = 0;
    }
    
    half4 col1 = (half4(spectralColorArray[int(colorIndex)].color[0]) *
        (1.0 - rem)) + (rem * half4(spectralColorArray[colorIndexAdjacent].color[0]));
    
    half4 col2 = (half4(spectralColorArray[int(colorIndex)].color[1]) *
        (1.0 - rem)) + (rem * half4(spectralColorArray[colorIndexAdjacent].color[1]));
    
    half4 col3 = (half4(spectralColorArray[int(colorIndex)].color[2]) *
        (1.0 - rem)) + (rem * half4(spectralColorArray[colorIndexAdjacent].color[2]));
    
//    if (logEncoding == 1) {
    wheelTexture.write(min( ((col1 + blackness) * (1.0 - greyRatio) + greyness ) * whiteness, 0.0), gid, 0);
    wheelTexture.write(min( ((col2 + blackness) * (1.0 - greyRatio) + greyness ) * whiteness, 0.0), gid, 1);
    wheelTexture.write(min( ((col3 + blackness) * (1.0 - greyRatio) + greyness ) * whiteness, 0.0), gid, 2);
//    } else {
//        wheelTexture.write((col1 + pow((1.0 - satSlider), 2.5)) * dist + grey, gid, 0);
//        wheelTexture.write((col2 + pow((1.0 - satSlider), 2.5)) * dist + grey, gid, 1);
//        wheelTexture.write((col3 + pow((1.0 - satSlider), 2.5)) * dist + grey, gid, 2);
//    }
    
    wheelTexture.write(half4(1,1,1,1), gid, 3);
    
}

static half drawEllipseForDab(float2 center, const constant Dab *dabArray, const constant DabMeta *dabMeta, uint2 gid, int dabIndex, half radius, half xOffset, half yOffset) {
    half dist;
    // draw ellipse and/or squircle shape
    if ( dabArray[dabIndex].dabRatio < 1.0 || dabArray[dabIndex].dabShape > 0.0) {
        // major and minor axis lengths
        half a = pow((radius * half(dabArray[dabIndex].dabRatio)), half(2.0));
        half b = pow((radius), half(2.0));
        
        half rotation = dabArray[dabIndex].dabAngle;
        
        // apply affine transform for rotation and offset
        float x = xOffset * cos(rotation) + (yOffset) * sin(rotation);
        float y = -1 * ((xOffset) * sin(rotation)) + (yOffset) * cos(rotation);
        
        // standard equation for ellipse == 1 if point is exactly on the ellipse perimeter
        half distEllipse = (pow(x, 2.0) / a ) + (pow((y), (2.0)) / b);
        
        // squircle
        half n = dabArray[dabIndex].dabShapeMod;
        half distSquircle = max(half(0.0), half(pow(half(abs(x / (radius * dabArray[dabIndex].dabRatio))), n) + pow(half(abs(y / radius)), n)));
        
        // interpolate between ellipse and squircle shape
        dist = (1.0 - dabArray[dabIndex].dabShape) * distEllipse + dabArray[dabIndex].dabShape * distSquircle;
    } else {
        // optimization for just a simple circle
        dist = distance(float2(gid) + float2(dabMeta->texOrigin), center) / radius;
    }
    return dist;
}



// draw a normal dab
static void drawNormalDab(const constant Dab *dabArray, int dabIndex, const constant DabMeta *dabMeta, thread half4 &dstA, thread half4 &dstB, thread half4 &dstC, thread half4 &dstMeta, texture2d_array <half, access::read> lowerLayer, thread const texture2d_array<half, access::read> &smudgeBuckets, uint2 gid, uint tid) {
    
    // center of the dab to draw
    float2 center = (dabArray[dabIndex].pos);
    // translate the position using the global texOrigin coordinates
    half xOffset = half(gid.x) + dabMeta->texOrigin.x - center.x;
    half yOffset = half(gid.y) + dabMeta->texOrigin.y - center.y;
    
    // coverage is either on or off for a pixel and is a % chance basically
    // needs seeds to avoid obvious repeating noise
    if (dabArray[dabIndex].dabCoverage < 1.0) {
        half dabCoverageChance = rand(xOffset + tid, yOffset + tid, dabMeta->randomSeeds2.y);
        if ( dabCoverageChance > dabArray[dabIndex].dabCoverage ) return;
    }
    
    // ratio of smudge bucket color to the brush color
    // 1.0 will smear existing paint
    half smudgeAmount = dabArray[dabIndex].smudgeAmount;
    
    // opacity is opaqueness
    // Partially opaque and thick paint will be darker
    half opacity = clamp(half(dabArray[dabIndex].opacity),EPSILON, half(1.0));
    // volume AKA thickness. Scale to 10.0 units
    half volume = clamp(half(dabArray[dabIndex].volume),EPSILON, half(10.0));
    // strength is ultimately the ratio of brush to existing canvas ratio, even
    // when erasing.  It's the strength of the effect of whatever you're doing.
    half strength = dabArray[dabIndex].strength;
    // radius of the dab to draw, in pixels
    half radius = dabArray[dabIndex].radius;
    // hardness is how much to feather the edge of a dab
    half hardness = dabArray[dabIndex].hardness;
    // eraser removes existing paint and thickness
    half eraser = 1.0 - (dabArray[dabIndex].eraser);
    
    // jitter the opacity
    if (dabArray[dabIndex].opacityJitterChance > 0.0 && dabArray[dabIndex].opacityJitter > 0.0) {
        half opacityJitterChance = rand(xOffset + tid, yOffset + tid, dabMeta->randomSeeds.x);
        if ( opacityJitterChance < dabArray[dabIndex].opacityJitterChance) opacity *= max(half(1.0 - dabArray[dabIndex].opacityJitter), opacityJitterChance);
    }
    
    // jitter smudge
    if (dabArray[dabIndex].smudgeJitterChance > 0.0 && dabArray[dabIndex].smudgeJitter > 0.0) {
        half smudgeJitterChance = rand(yOffset + tid, xOffset + tid, dabMeta->randomSeeds.y);
        if (( smudgeJitterChance < dabArray[dabIndex].smudgeJitterChance )) smudgeAmount *= max(half(1.0 - dabArray[dabIndex].smudgeJitter), smudgeJitterChance);
    }
    
    // use a signed distance field to draw the shape
    half dist = drawEllipseForDab(center, dabArray, dabMeta, gid, dabIndex, radius, xOffset, yOffset);
    
    // if outside the ellipse, don't draw anything for this dab
    // otherwise, use the distance to adjust strength to fade out w/ hardness parameter
    dist = pow(dist, half(30.0) * hardness);
    if (dist >= 1.0 || dist < 0.0 || isnan(dist)) return;

    strength *= (1.0 - dist);
    
    
    half strengthInv = (1.0 - strength);
    half eraserStrength = 1.0 - (dabArray[dabIndex].eraser * strength);
    
    strength *= eraser;
    
    
    
    // Smudge Bucket is just a small texture that stores
    // a bunch of colors that can be recalled on a per-dab
    // basis. Kind of good for matching bristles to their
    // own smudge color
    uint2 bucket = uint2(dabArray[dabIndex].smudgeBucket, 0);
    uint2 bucketMeta = uint2(dabArray[dabIndex].smudgeBucket, 1);
    half4 smudgeBucketMeta = smudgeBuckets.read(bucketMeta, 0);
    
    // ensure smudge bucket has been initialized and sampled at least once
    if (smudgeBucketMeta.y < 1 && smudgeAmount > 0) {
        return;
    }
    half4 smudgeBucketD = smudgeBuckets.read(bucket, 3);
    
    

    
    
    half invSmudgeAmount = (1.0 - smudgeAmount);
    
    // jitter the volume/thickness
    if (dabArray[dabIndex].volumeJitterChance > 0.0) {
        half volumeJitterChance = rand(yOffset + tid, xOffset + tid, dabMeta->randomSeeds2.x);
        if (( volumeJitterChance < dabArray[dabIndex].volumeJitterChance )) volume *= max(half(1.0 - dabArray[dabIndex].volumeJitter), volumeJitterChance);
    }
    
    
    half4 smudgeBucketA = smudgeBuckets.read(bucket, 0);
    half4 smudgeBucketB = smudgeBuckets.read(bucket, 1);
    half4 smudgeBucketC = smudgeBuckets.read(bucket, 2);
    
    
    // if there is a layer below the active layer, we might want
    // to interact with it somehow (read-only). Here we lift paint
    // up from the lower layer and mix it with the smudge color
    // but only if our paint is a solvent and can disolve it
    if (dabMeta->hasLowerTexture == 1) {
        half4 lowerA = lowerLayer.read(gid, 0);
        half4 lowerB = lowerLayer.read(gid, 1);
        half4 lowerC = lowerLayer.read(gid, 2);
        half4 lowerMeta = lowerLayer.read(gid, 3);
        
        half liftFac = clamp((lowerMeta.y / 10.0) * dabArray[dabIndex].solvent, 0.0, 1.0);
        smudgeBucketA = smudgeBucketA * (1.0 - liftFac) + lowerA * liftFac;
        smudgeBucketB = smudgeBucketB * (1.0 - liftFac) + lowerB * liftFac;
        smudgeBucketC = smudgeBucketC * (1.0 - liftFac) + lowerC * liftFac;
        smudgeBucketD = smudgeBucketD * (1.0 - liftFac) + lowerMeta * liftFac;
    }
    
    
    // calculate volume/thickness
    half volumeTop = invSmudgeAmount * volume;
    half volumeBottom = smudgeAmount * smudgeBucketD.y;
    half volumeResult = (strength * (volumeTop + volumeBottom)) + eraserStrength * strengthInv * dstMeta.y;
    volumeResult = clamp(volumeResult, half(0.0), half(10.0));
    
    // this is less weird than smudgeThicknessThreshold.
    // Don't paint if the brush thickness is going to add less than what is already on the canvas
    half thicknessThreshold = dabArray[dabIndex].thicknessThreshold * eraser;
    if (volumeTop < thicknessThreshold * dstMeta.y) {
        return;
    }
    
    
    
    
    
    
    // calculate opacity before applying thickness
    half topOpacity = (smudgeAmount * smudgeBucketD.x + invSmudgeAmount * opacity) * strength;
    half bottomOpacity = strengthInv * dstMeta.x;
    half opacityResult = (topOpacity + bottomOpacity);
    
    // apply beer-lambert-like multiplier to the color based on opacity and thickness
    half beerMultiplier = (volume * ((half(1.0) - opacity))) + 1.0;
    
    
    half wetness = eraserStrength * clamp(half(strength * (1.0 - dabArray[dabIndex].wetness) +  strengthInv * dstMeta.w), half(0.0), half(1.0));
    half beerMultiplierTop = (smudgeAmount * smudgeBucketD.z + invSmudgeAmount * beerMultiplier) * strength;
    half beerResult = (beerMultiplierTop + strengthInv * dstMeta.z);
    
    // brush color, 12 channels, remember?
    // log2 encoded
    half4 color0 = half4(dabArray[dabIndex].color[0]);
    half4 color1 = half4(dabArray[dabIndex].color[1]);
    half4 color2 = half4(dabArray[dabIndex].color[2]);
    
    
    // jitter the color channels left or right
    if ( dabArray[dabIndex].colorJitter != 0.0 && dabArray[dabIndex].colorJitterChance > 0.0) {
        
        half colorJitterChance = rand(yOffset + tid, xOffset + tid, dabMeta->randomSeeds.w);
        half colorJitter = dabArray[dabIndex].colorJitter * colorJitterChance;
        
        uint iterations = abs(colorJitter) * 24.0 + 1.0;
        
        if (colorJitterChance < dabArray[dabIndex].colorJitterChance) {
            if (colorJitter > 0.0) {
                
                half invColorJitter = 1.0 - colorJitter;
                for (uint iter = 0; iter < iterations; iter++) {
                    for (uint i=0; i < 3; i++) {
                        color0[i] = color0[i] * invColorJitter + colorJitter * color0[i + 1];
                        color1[i] = color1[i] * invColorJitter + colorJitter * color1[i + 1];
                        color2[i] = color2[i] * invColorJitter + colorJitter * color2[i + 1];
                    }
                    color0[3] = color0[3] * invColorJitter + colorJitter * color1[0];
                    color1[3] = color1[3] * invColorJitter + colorJitter * color2[0];
                    color2[3] = color2[3] * invColorJitter + colorJitter * color0[0];
                }
                
            } else if (colorJitter < 0.0) {
                colorJitter = -colorJitter;
                half invColorJitter = 1.0 - colorJitter;
                for (uint iter = 0; iter < iterations; iter++) {
                    for (uint i=3; i > 0; i--) {
                        color0[i] = color0[i] * invColorJitter + colorJitter * color0[i - 1];
                        color1[i] = color1[i] * invColorJitter + colorJitter * color1[i - 1];
                        color2[i] = color2[i] * invColorJitter + colorJitter * color2[i - 1];
                    }
                    color0[0] = color0[0] * invColorJitter + colorJitter * color2[3];
                    color1[0] = color1[0] * invColorJitter + colorJitter * color0[3];
                    color2[0] = color2[0] * invColorJitter + colorJitter * color1[3];
                }
            }
        }
    }
    
    if ( dabArray[dabIndex].valueJitter != 0.0 && dabArray[dabIndex].valueJitterChance > 0.0) {
        half valueJitterChance = 1.0 - rand(xOffset + tid, yOffset + tid, dabMeta->randomSeeds.y);
        half valueJitter =  dabArray[dabIndex].valueJitter * valueJitterChance;
        
        if (valueJitterChance <= dabArray[dabIndex].valueJitterChance) {
            if (valueJitter < 0.0 ) {
                valueJitter = -valueJitter;
                valueJitter = pow(1.0 - valueJitter, 4.0);
                for (uint i=0; i < 4; i++) {
                    color0[i] = max(EPSILON_LOG, color0[i] / (valueJitter));
                    color1[i] = max(EPSILON_LOG, color1[i] / (valueJitter));
                    color2[i] = max(EPSILON_LOG, color2[i] / (valueJitter));
                    
                }
            } else {
                valueJitter = pow(1.0 - valueJitter, 4.0);
                
                for (uint i=0; i < 4; i++) {
                    color0[i] *= valueJitter;
                    color1[i] *= valueJitter;
                    color2[i] *= valueJitter;
                }
            }
        }
    }
    
    // combine smudge, brush color,
    half4 colorA = (smudgeAmount * smudgeBucketA + invSmudgeAmount * color0 * beerMultiplier) * strength;
    half4 colorB = (smudgeAmount * smudgeBucketB + invSmudgeAmount * color1 * beerMultiplier) * strength;
    half4 colorC = (smudgeAmount * smudgeBucketC + invSmudgeAmount * color2 * beerMultiplier) * strength;
    
    dstA = colorA + strengthInv * dstA;
    dstB = colorB + strengthInv * dstB;
    dstC = colorC + strengthInv * dstC;
    dstMeta = half4(clamp(opacityResult, half(0.0), half(1.0)), volumeResult, beerResult, wetness);
}

// non-kernel function to handle drawing dabs
// this lets us call this function from another kernel function (custom brush shader, for example)

static void normalBrush(constant Dab *dabArray [[ buffer(0) ]],
                     constant DabMeta *dabMeta [[ buffer(1) ]],
                     texture2d_array <half, access::read_write> activeLayer [[texture(0)]],
                     texture2d_array <half, access::read> smudgeBuckets [[texture(1)]],
                     texture2d_array <half, access::read> lowerLayer [[texture(2)]],
                     texture2d_array <half, access::sample> activeLayerSampler [[texture(3)]],
                     uint2 gid [[thread_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]]) {
    
    // read the active layer pixels into dstX
    // we will modify this data repeatedly for each dab and write it back into the layer at the end
    // our data format is 16 channels. 12 log2 color channels (first 3 textures) and a metadata texture
    // metadata stores opacity, thickness, a thickness/opacity factor, and a "worked" factor that negates the
    // background texture tooth effects
    half4 dstA = activeLayer.read(gid, 0);
    half4 dstB = activeLayer.read(gid, 1);
    half4 dstC = activeLayer.read(gid, 2);
    half4 dstMeta = activeLayer.read(gid, 3);
    
//    half4 lowerA, lowerB, lowerC, lowerMeta;
//    if (dabMeta->hasLowerTexture == 1) {
//        // if there is a lower layer, read that in. We might use it
//
//        lowerA = lowerLayer.read(gid, 0);
//        lowerB = lowerLayer.read(gid, 1);
//        lowerC = lowerLayer.read(gid, 2);
//        lowerMeta = lowerLayer.read(gid, 3);
//    }
    
    // for each dab, do a bunch of stuff and store it in the dst
    int dabCount = dabMeta->dabCount;
    for (int dabIndex=0; dabIndex < dabCount; dabIndex++) {
        drawNormalDab(dabArray, dabIndex, dabMeta, dstA, dstB, dstC, dstMeta, lowerLayer, smudgeBuckets, gid, tid);
    }
    
    if (any(isnan(dstA)) || any(isnan(dstA))
        || any(isinf(dstB)) || any(isinf(dstB))
        || any(isinf(dstC)) || any(isinf(dstC))
        || any(isinf(dstMeta)) || any(isinf(dstMeta))
        ) {
        return;
    }
    activeLayer.write(dstA, gid, 0);
    activeLayer.write(dstB, gid, 1);
    activeLayer.write(dstC, gid, 2);
    activeLayer.write(dstMeta, gid, 3);
}


kernel void drawDabs(constant Dab *dabArray [[ buffer(0) ]],
                    constant DabMeta *dabMeta [[ buffer(1) ]],
                    texture2d_array <half, access::read_write> activeLayer [[texture(0)]],
                    texture2d_array <half, access::read> smudgeBuckets [[texture(1)]],
                    texture2d_array <half, access::read> lowerLayer [[texture(2)]],
                    texture2d_array <half, access::sample> activeLayerSampler [[texture(3)]],
                    uint2 gid [[thread_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {

    normalBrush(dabArray, dabMeta, activeLayer, smudgeBuckets, lowerLayer, activeLayerSampler, gid, tid);

}



static void overOperation(texture2d_array<half, access::read> src [[texture(0)]],
                   texture2d_array<half, access::read_write> dst [[texture(1)]],
                   constant OverOp &overOp [[ buffer(0) ]],
                   uint2 gid [[thread_position_in_grid]]
                   ) {
    half4 srcS = src.read(gid, 0);
    half4 srcM = src.read(gid, 1);
    half4 srcL = src.read(gid, 2);
    half4 srcMeta = src.read(gid, 3);
    half4 dstS = dst.read(gid, 0);
    half4 dstM = dst.read(gid, 1);
    half4 dstL = dst.read(gid, 2);
    half4 dstMeta = dst.read(gid, 3);
    
    half srcAndLayerOpacity = srcMeta.x * overOp.srcOpacity * overOp.srcThickness ; // src opacity * slider value
    half dstAndLayerOpacity = (1.0 - srcAndLayerOpacity) * overOp.dstThickness; // dst opaicty * slider value
    half srcThickness = srcMeta.y * overOp.srcThickness; // src * slider value
    half dstThickness = dstMeta.y * overOp.dstThickness; // dst * slider value
    
    srcS = min(srcS, 0.0) * overOp.srcThickness + dstAndLayerOpacity * min(dstS, 0.0);
    srcM = min(srcM, 0.0) * overOp.srcThickness + dstAndLayerOpacity * min(dstM, 0.0);
    srcL = min(srcL, 0.0) * overOp.srcThickness + dstAndLayerOpacity * min(dstL, 0.0);

    half opacity = srcAndLayerOpacity + dstMeta.x * dstAndLayerOpacity;// - (srcAndLayerOpacity * dstAndLayerOpacity);
    // max out volume at 10.0 so that things aren't bumpy forever w/ many layers
    half volume = clamp((srcThickness + dstThickness), half(0.0), half(10.0));

    
    half beerFac = (srcMeta.z * overOp.srcThickness + dstAndLayerOpacity * dstMeta.z);
    half wetness = (srcMeta.w * srcAndLayerOpacity + dstAndLayerOpacity * dstMeta.w);
    srcMeta = half4(clamp(opacity, half(0.0), half(1.0)), volume, beerFac, wetness);
    
    dst.write(srcS, gid, 0);
    dst.write(srcM, gid, 1);
    dst.write(srcL, gid, 2);
    dst.write(srcMeta, gid, 3);
}


kernel void spectralOver(texture2d_array<half, access::read> src [[texture(0)]],
                 texture2d_array<half, access::read_write> dst [[texture(1)]],
                 constant OverOp &overOp [[ buffer(0) ]],
                 uint2 gid [[thread_position_in_grid]]) {
    overOperation(src, dst, overOp, gid);
    
}

// Vertex Function
vertex RasterizerData
vertexShader(uint vertexID [[ vertex_id ]],
             constant Vertex *vertexArray [[ buffer(0) ]],
             constant vector_float2 *viewportSizePointer  [[ buffer(1) ]])

{

    RasterizerData out;
    
    float2 pixelSpacePosition = vertexArray[vertexID].pos.xy;

    float2 viewportSize = float2(*viewportSizePointer);
    out.clipSpacePosition.xy = pixelSpacePosition / (viewportSize / 2.0);
    out.clipSpacePosition.z = 0.0;
    out.clipSpacePosition.w = 1.0;
    out.textureCoordinate = vertexArray[vertexID].textureCoordinate;

    return out;
}

fragment float4 samplingShader(RasterizerData in [[stage_in]],
                               texture2d<float> colorTexture [[ texture(0) ]])
{
    constexpr sampler textureSampler (mag_filter::bicubic,
                                      min_filter::bicubic);
    const float4 colorSample = colorTexture.sample (textureSampler, in.textureCoordinate);
    return float4(colorSample);
}





//CUSTOM_SHADER_BEGIN
