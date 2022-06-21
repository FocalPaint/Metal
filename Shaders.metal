//
//  Shaders.metal
//
//  Created by brien dieterle on 5/24/20.
//  Copyright © 2020 brien dieterle. All rights reserved.
//

#include <metal_stdlib>
#include "ShaderDefinitions.h"
using namespace metal;


constant half3 displayP3Luma = half3(0.265667693, 0.691738522, 0.0451133819);
constant half EPSILON = 0.0002; // for some reason 0.0001 is too small, NaNs
constant half offset = 1.0 - EPSILON;
constant half EPSILON_LOG = -12.287712379549449;

constant half T_MATRIX [3][12] ={{0.043360489449518, 0.026456474761540, 0.020238299013514,
    -0.074764289932655, -0.170370642119887, -0.104586664428807,
    -0.086581468179310, -0.068906811339458, 0.457056404183395,
    0.437453976232458, 0.431715004832538, 0.157869804779479},
    {-0.040534472903886, -0.033419552329132, -0.029293045613806,
    0.057466135999207, 0.192345371870075, 0.305415920318147,
    0.293190243033650, 0.282447833359094, -0.000701733209694,
        -0.015668249622555, -0.018679051015265, -0.015331520111816},
    {0.309776255557529, 0.356528781924368, 0.344528004764383,
    0.114183770252453, 0.016619897453525, -0.032036721687686,
    -0.031625939891442, -0.031257924571711, -0.009110338431730,
    -0.006870381924116, -0.006379347447639, -0.001282228702366}};

constant half4 redShort = {0.000102347885217, 0.000101450240849, 0.000102203595088,
    0.000182068610178};
constant half4 redMedium = { 0.000335278671537, 0.000107101988920,
    0.000235302112089, 0.000152926173608 };
constant half4 redLong = { 0.678406822261829,
    0.963465416215934, 0.925221300535300, 0.595455091698738 };

constant half4 greenShort = { 0.000100517500654, 0.000100142149336, 0.000100230124224,
    0.000100577314601};
constant half4 greenMedium = { 0.979913754835688, 0.975279095060780,
    0.979951868675621, 0.970116202772815 };
constant half4 greenLong = { 0.274236729494797,
    0.000238846302870, 0.031458970725383, 0.371961283042717 };

constant half4 blueShort = { 0.979983032675131, 0.979938629678512, 0.979928904521242,
    0.979683076599390 };
constant half4 blueMedium = { 0.019743831047335, 0.054752823745413,
    0.037082288916508, 0.047894730049648 };
constant half4 blueLong = { 0.000100410422990,
    0.000100262330692, 0.000101046193637, 0.002318893233423};


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
                          uint2 gid [[thread_position_in_grid]]) {
    half4 rgba = rgbTexture.read(gid);
    half4 greyRGBA = greyTexture.read(gid);
    //half grey = dot(greyRGBA.rgb, displayP3Luma);
    half alpha = rgba.w;
    if (alpha > 0.0) {
        rgba /= alpha;
    }
    rgba = clamp(rgba, 0.0, 1.0);
    rgba = (rgba * offset) + EPSILON;
    
    // if greyscale, just use that value for all channels
    // otherwise use color
    
    half4 colorShort;
    half4 colorMedium;
    half4 colorLong;
    if ( rgba.x == rgba.y == rgba.z ) {
        colorShort = half4(rgba.x, rgba.x, rgba.x, rgba.x );
        colorMedium = half4(rgba.x, rgba.x, rgba.x, rgba.x );
        colorLong = half4(rgba.x, rgba.x, rgba.x, rgba.x );
    } else {
        colorShort = redShort * rgba.x + greenShort * rgba.y + blueShort * rgba.z;
        colorMedium = redMedium * rgba.x + greenMedium * rgba.y + blueMedium * rgba.z;
        colorLong = redLong * rgba.x + greenLong * rgba.y + blueLong * rgba.z;
    }
    
    spectralTexture.write(log2(colorShort) * alpha, gid, 0);
    spectralTexture.write(log2(colorMedium) * alpha, gid, 1);
    spectralTexture.write(log2(colorLong) * alpha, gid, 2);
    spectralTexture.write(half4(alpha, greyRGBA.r, 0.0, 0.0), gid, 3);
    
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
                          uint2 gid [[thread_position_in_grid]]) {

    // read spectral array texture and write to an RGB texture with associated alpha
    
    half4 srcS = spectralTexture.read(gid, 0);
    half4 srcM = spectralTexture.read(gid, 1);
    half4 srcL = spectralTexture.read(gid, 2);
//    half4 srcMeta = spectralTexture.read(gid, 3);
    
    half4 S = (srcS);
    half4 M = (srcM);
    half4 L = (srcL);
    half spd [12] = {S.x, S.y, S.z, S.w, M.x, M.y, M.z, M.w, L.x, L.y, L.z, L.w};
    
    half3 rgb = {0};
    
    // convert back to RGB
    for (int i=0; i<12; i++) {
        rgb[0] += T_MATRIX[0][i] * spd[i];
        rgb[1] += T_MATRIX[1][i] * spd[i];
        rgb[2] += T_MATRIX[2][i] * spd[i];
    }
    
    
    // undo offset
    
    half3 resColor = ((rgb - EPSILON) / offset);
//    half alpha = srcMeta.x;
//    if (alpha > 0.0 && alpha < 1.0) {
//
//        resColor /= alpha;
//    }
    
    resColor = saturate(resColor);
    
    // apply alpha to pixels so that white=black, invert for normal programs.  Only means anything with background disabled
    half4 renderedPixel = half4(resColor, 1.0);
    rgbTexture.write(renderedPixel, gid); //write to render target

    
    
}

kernel void spectralLogToRGB(texture2d_array <half, access::read> spectralTexture [[texture(0)]],
                          texture2d <half, access::read_write> rgbTexture [[texture(1)]],
                          uint2 gid [[thread_position_in_grid]]) {

    // read spectral array texture and write to an RGB texture with associated alpha
    
    half4 srcS = spectralTexture.read(gid, 0);
    half4 srcM = spectralTexture.read(gid, 1);
    half4 srcL = spectralTexture.read(gid, 2);
//    half4 srcMeta = spectralTexture.read(gid, 3);
//    half alpha = srcMeta.x;
//    if (alpha <= 0.0) {
//
//        alpha = 1.0;
//    }
    
    // log to linear
    half4 S = exp2(srcS);
    half4 M = exp2(srcM);
    half4 L = exp2(srcL);
    half spd [12] = {S.x, S.y, S.z, S.w, M.x, M.y, M.z, M.w, L.x, L.y, L.z, L.w};
    
    half3 rgb = {0};
    
    // convert back to RGB
    for (int i=0; i<12; i++) {
        rgb[0] += T_MATRIX[0][i] * spd[i];
        rgb[1] += T_MATRIX[1][i] * spd[i];
        rgb[2] += T_MATRIX[2][i] * spd[i];
    }
    
    
    // undo offset
    
    half3 resColor = ((rgb - EPSILON) / offset);
    resColor = saturate(resColor);
//    if (alpha > 0.0 && alpha < 1.0) {
//
//        resColor /= alpha;
//    }
    
    // apply alpha to pixels so that white=black, invert for normal programs.  Only means anything with background disabled
    half4 renderedPixel = half4(resColor, 1.0);
    rgbTexture.write(renderedPixel, gid); //write to render target

    
    
}




kernel void updateSmudgeBuckets(constant Dab *dabArray [[ buffer(0) ]],
                    constant DabMeta *DabMeta [[ buffer(1) ]],
                    texture2d_array <half, access::read> canvas [[texture(0)]],
                    texture2d_array <half, access::read_write> smudgeBuckets [[texture(1)]],
                    uint2 gid [[thread_position_in_grid]]) {


    int dabCount = DabMeta->dabCount;
    
    
    //int bucketCount = DabMeta->bucketCount - 1;

    for (int i=0; i < dabCount; i++) {
        
        half smudgeLength = dabArray[i].smudgeLength;
        if (smudgeLength >= 1.0) continue;
        
        half2 center = half2(dabArray[i].pos);
        half dist = distance(half2(gid) + half2(DabMeta->texOrigin), center);
        if (dist > dabArray[i].smudgeRadius) continue; // skip sampling beyond the smudge radius
        
        // bucket to update/average into
        uint2 bucket = uint2(dabArray[i].smudgeBucket, 0);
        
        half4 smudgeBucketD = smudgeBuckets.read(bucket, 3);
        // how recent this bucket was updated, break early
        half recentness = smudgeBucketD.w * smudgeLength;
        
        // reset recentness when we sample
        if (recentness < 0.5) {
            recentness = 1.0;
            
            if (recentness == 0.0) {
                smudgeLength = 0.0;
            }
            
        } else {
            // use recentness for w channel instead of "worked"
            smudgeBuckets.write(half4(smudgeBucketD.x, smudgeBucketD.y, smudgeBucketD.z, recentness), bucket, 3);
            smudgeBuckets.fence();
            continue;
        }
        
        half4 smudgeBucketA = smudgeBuckets.read(bucket, 0);
        half4 smudgeBucketB = smudgeBuckets.read(bucket, 1);
        half4 smudgeBucketC = smudgeBuckets.read(bucket, 2);
        
        // sample the canvas
        half4 smudgeSampleA = 0;
        half4 smudgeSampleB = 0;
        half4 smudgeSampleC = 0;
        half4 smudgeSampleD = 0;


        smudgeSampleA = canvas.read(gid, 0);
        smudgeSampleB = canvas.read(gid, 1);
        smudgeSampleC = canvas.read(gid, 2);
        smudgeSampleD = canvas.read(gid, 3);
        
        smudgeBucketA = smudgeBucketA * smudgeLength + (1.0 - smudgeLength) * smudgeSampleA;
        smudgeBucketB = smudgeBucketB * smudgeLength + (1.0 - smudgeLength) * smudgeSampleB;
        smudgeBucketC = smudgeBucketC * smudgeLength + (1.0 - smudgeLength) * smudgeSampleC;
        smudgeBucketD = smudgeBucketD * smudgeLength + (1.0 - smudgeLength) * smudgeSampleD;
         
        smudgeBuckets.write(smudgeBucketA, bucket, 0);
        smudgeBuckets.write(smudgeBucketB, bucket, 1);
        smudgeBuckets.write(smudgeBucketC, bucket, 2);
        
        // use recentness for w channel instead of "worked"
        smudgeBuckets.write(half4(smudgeBucketD.x, smudgeBucketD.y, smudgeBucketD.z, recentness), bucket, 3);
        smudgeBuckets.fence();
        
    }

}



kernel void applyBumpMap(texture2d_array <half, access::read_write> canvas [[texture(0)]],
                         //constant float2 &bumpOpts [[ buffer(0) ]],
                         uint2 gid [[thread_position_in_grid]]) {
    

    //half4 cMeta = canvas.read(gid, 3);
    //if (cMeta.x <= 0.0) return;
    
    //half alpha = cMeta.x;
    
    // log to linear
    
    // normalize then reassociate alpha after we're back in linear
    
    half4 cS = exp2(canvas.read(gid, 0));
    half4 cM = exp2(canvas.read(gid, 1));
    half4 cL = exp2(canvas.read(gid, 2));
    
    half Gx = 0;
    half Gy = 0;

    // East
    Gx -= canvas.read(gid - uint2(1, 0), 3).y * 2;

    // West
    Gx += canvas.read(gid + uint2(1, 0), 3).y * 2;
    
    // North-East
    Gx -= canvas.read(gid - uint2(0, 1) + uint2(1, 0), 3).y;
    
    // North-West
    Gx += canvas.read(gid - uint2(1, 1), 3).y;
    
    // South-East
    Gx -= canvas.read(gid + uint2(1, 1), 3).y;
    
    // South-West
    Gx += canvas.read(gid + uint2(0, 1) - uint2(1, 0), 3).y;
    
    

    // North
    Gy -= canvas.read(gid - uint2(0, 1), 3).y * 2;

    // South
    Gy += canvas.read(gid + uint2(0, 1), 3).y * 2;
    
    // North-East
    Gy -= canvas.read(gid - uint2(0, 1) + uint2(1, 0), 3).y;

    // North-West
    Gy -= canvas.read(gid - uint2(1, 1), 3).y;

    // South-East
    Gy += canvas.read(gid + uint2(1, 1), 3).y;

    // South-West
    Gy += canvas.read(gid + uint2(0, 1) - uint2(1, 0), 3).y;
    
    
    
    const half Oren_rough = 0.5 ; //clamp(1.0 - cMeta.z, 0.0, 1.0);
    const half Oren_A = 1.0 - 0.5 * (Oren_rough / (Oren_rough + 0.33));
    const half Oren_B = 0.45 * (Oren_rough / (Oren_rough + 0.09));
    const half Oren_exposure = 1.0 / Oren_A; // dumb hack to avoid darkening
    //const half amp = cMeta.z / (cMeta.x > 0.0 ? cMeta.x : 1.0) + cMeta.y; // normalize bumpamp before using

    const half slope = sqrt(Gx * Gx + Gy * Gy);
    //slope = slope;// * amp;
    //slope = clamp(amp * float(slope), 0.0, 2.0);
    const half radians = atan2(Gx, Gy);
    //half direction = smallest_angular_difference(radians * 180.0f / M_PI_F, 60.0) ;
    const half direction = min((2 * M_PI_H) - abs(radians - 1), abs(radians - 1)) / M_PI_F; // direction normalized, 1 when SW, 0 when NE
    const half degrees = atan(slope * direction);
    const half specular = direction < 0.2 ? atan((1.0 - direction) * slope) : 0;
    const half lambert = clamp((cos(degrees) * (Oren_A + (Oren_B * sin(degrees) * tan(degrees)))) * Oren_exposure, half(0.0), half(1.0)) * (1.0 + specular * 0.2);

    if (lambert != 0.0) {
        cS *= lambert;
        cM *= lambert;
        cL *= lambert;

    }
    
    canvas.write(cS, gid, 0);
    canvas.write(cM, gid, 1);
    canvas.write(cL, gid, 2);
       // }
}



// reduce the paint of dst based on jaggies of src

kernel void reducePaint(texture2d_array <half, access::read> src [[texture(0)]],
                        texture2d_array <half, access::read_write> dst [[texture(1)]],
                        //constant DabMeta *DabMeta [[ buffer(0) ]],
                        //constant float2 &bumpOpts [[ buffer(0) ]],
                        uint2 gid [[thread_position_in_grid]]) {
    
    half4 dstS = dst.read(gid, 0);
    half4 dstM = dst.read(gid, 1);
    half4 dstL = dst.read(gid, 2);
    half4 dstMeta = dst.read(gid, 3);
    //half volume = dstMeta.x > 0.0 ? dstMeta.y / dstMeta.x : dstMeta.y;
    //half depth =  clamp(src.read(gid, 3).y + half(dstMeta.y / 2.0) + half(dstMeta.w / 10.0), half(0.0), half(1.0));
    half depth =  clamp(src.read(gid, 3).y * half(10.0) + half(dstMeta.y / 5.0) + half(dstMeta.w / 10.0), half(0.0), half(1.0));
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
    dst.write(half4(fillColor->color[1]) + dstL, gid, 2);
    
    
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
    
    if (logEncoding == 1) {
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
    
    if (logEncoding == 1) {
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


// example custom shader, this "pushes" paint around based on the dab angle, which you can
// control by mapping Stroke Direction or Azimuth, or whatever, to control.

void blendCustom(half4 middle[4], half4 leading[4], half4 trailing[4], half strength)
{
    
    
    //if ( leading[3].y < center[3].y * 0.1 || trailing[3].y < center[3].y * 0.1 ) return;

    //half vol = middle[3].y;
    float strengthForward = clamp(strength + (float(middle[3].y  - leading[3].y ) / (middle[3].y + leading[3].y)), 0.0, 1.0);
    float strengthInvF=(1.0 - strengthForward);

    half volIncrease = (trailing[3].y < 1 && middle[3].y > 0.0 && middle[3].y < 10.0) ?  strengthForward + 1.0 : 1.0 ;
//    strengthInvF = pow( (strengthInvF), 1.0 / volIncrease);
    


    middle[3].x = clamp(middle[3].x * strengthInvF + leading[3].x * strengthForward, 0.0, 1.0);
    
    half beerFacMiddle = middle[3].z > 0.0 ? middle[3].z : 1.0;
    half beerFacLeading = middle[3].z > 0.0 ? middle[3].z : 1.0;

    middle[3].y = clamp(middle[3].y * pow((strengthInvF), 1.0 / volIncrease) + leading[3].y * strengthForward, 0.0, 10.0);
    middle[3].w = middle[3].w * strengthInvF + leading[3].w * strengthForward;
    middle[3].w = clamp( middle[3].w + strength * 0.01, 0.0, 1000.0);
    middle[3].z = middle[3].z * strengthInvF + leading[3].z * strengthForward;
    
    // apply beer-lambert-like multiplier to the color based on opacity and thickness
    half beerMultiplier = (middle[3].y * ((half(1.0) - middle[3].x))) + 1.0;

    // blend w/ canvas pixel
    for (int i = 0; i < 3; ++i) {
        middle[i] = ((middle[i] / beerFacMiddle) * strengthInvF + (leading[i] / beerFacLeading) * strengthForward) * beerMultiplier;
    }
   
}

kernel void customBrushShader(
  constant Dab *dabArray [[ buffer(0) ]],
  constant DabMeta *dabMeta [[ buffer(1) ]],
  texture2d_array <half, access::read_write> canvas [[texture(0)]],
  texture2d_array <half, access::read> smudge [[texture(1)]],
  texture2d_array <half, access::read> lowerCanvas [[texture(2)]],
  texture2d_array <half, access::sample> canvasSample [[texture(3)]],
  uint2 gid [[thread_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]])
  {
//      constexpr sampler s(coord::pixel,
//                    address::clamp_to_edge,
//                    filter::nearest);
      
//      if (gid.x == 0 || gid.y == 0 | gid.x >= canvas.get_width() - 1 || gid.y >= canvas.get_height() - 1) return;
      
      
//      half4 middle[4];
//      half4 left[4];
//      half4 upperLeft[4];
//      half4 up[4];
//      half4 upperRight[4];
//      half4 right[4];
//      half4 lowerRight[4];
//      half4 down[4];
//      half4 lowerLeft[4];
//
//      int i;
//      for (i = 0; i < 4; ++i) {
//          middle[i] = canvas.read(gid, i);
//          left[i] = canvas.read(uint2(int2(gid) - int2(1,0)), i);
//          upperLeft[i] = canvas.read(uint2(int2(gid) - int2(1,1)), i);
//          up[i] = canvas.read(uint2(int2(gid) - int2(0,1)), i);
//          upperRight[i] = canvas.read(uint2(int2(gid) - int2(-1,1)), i);
//          right[i] = canvas.read(uint2(int2(gid) - int2(-1,0)), i);
//          lowerRight[i] = canvas.read(uint2(int2(gid) - int2(-1,-1)), i);
//          down[i] = canvas.read(uint2(int2(gid) - int2(0,-1)), i);
//          lowerLeft[i] = canvas.read(uint2(int2(gid) - int2(1,-1)), i);
//      }
      
      // for each dab, do a bunch of stuff and store it in the dst
      int dabCount = dabMeta->dabCount;
      for (int i=0; i < dabCount; ++i) {
          // center of the dab to draw
          float2 center = (dabArray[i].pos);
          // translate the position using the global texOrigin coordinates
          half xOffset = half(gid.x) + dabMeta->texOrigin.x - center.x;
          half yOffset = half(gid.y) + dabMeta->texOrigin.y - center.y;
          half strength = dabArray[i].strength;
          // radius of the dab to draw, in pixels
          half radius = dabArray[i].radius;
          // hardness is how much to feather the edge of a dab
          half hardness = dabArray[i].hardness;
          
          half rotation = dabArray[i].dabAngle;
          
          // use a signed distance field to draw the shape
          half dist;
          // draw ellipse and/or squircle shape
          if ( dabArray[i].dabRatio < 1.0 || dabArray[i].dabShape > 0.0) {
              // major and minor axis lengths
              half a = pow((radius * half(dabArray[i].dabRatio)), half(2.0));
              half b = pow((radius), half(2.0));

              
              
              // apply affine transform for rotation and offset
              float x = xOffset * cos(rotation) + (yOffset) * sin(rotation);
              float y = -1 * ((xOffset) * sin(rotation)) + (yOffset) * cos(rotation);

              // standard equation for ellipse == 1 if point is exactly on the ellipse perimeter
              half distEllipse = (pow(x, 2.0) / a ) + (pow((y), (2.0)) / b);
              
              // squircle
              half n = dabArray[i].dabShapeMod;
              half distSquircle = max(half(0.0), half(pow(half(abs(x / (radius * dabArray[i].dabRatio))), n) + pow(half(abs(y / radius)), n)));

              // interpolate between ellipse and squircle shape
              dist = (1.0 - dabArray[i].dabShape) * distEllipse + dabArray[i].dabShape * distSquircle;
          } else {
              // optimization for just a simple circle
              dist = distance(float2(gid) + float2(dabMeta->texOrigin), center) / radius;
          }
          
          // if outside the ellipse, don't draw anything for this dab
          if (dist > 1.0 || dist < 0.0 || isnan(dist)) continue;
          // otherwise, use the distance to adjust strength to fade out w/ hardness parameter
          strength *= (1.0 - (pow(dist, half(30.0) * hardness)));
          
          // determine which two samples to read based on angle
          int2 dir = int2(round(normalize(float2(cos(rotation), sin(rotation))  )* 1.8));
          int2 dirRev = int2(-1.0 * float2(dir));
          //uint2 sample_coord_trailing = uint2(int2(gid) + int2(dir));

//          if (dir.x == -1 && dir.y == 0) {
          
          half4 middle[4];
          half4 leading[4];
          half4 trailing[4];
          for (int i = 0; i < 4; ++i) {
              middle[i] = canvas.read(gid, i);
              leading[i] = canvas.read(uint2(int2(gid) - dir), i);
              trailing[i] = canvas.read(uint2(int2(gid) - dirRev), i);
              
          }
          
          blendCustom(middle, leading, trailing, strength);
          for (int i = 0; i < 4; ++i) {
            canvas.write(middle[i], gid, i);
//            canvas.write(leading[i], uint2(int2(gid) - dir), i);
//            canvas.write(trailing[i], uint2(int2(gid) - dirRev), i);
//
          }
         
        
//          }
//
//
//          uint2 sample_coord_leading = uint2(float2(gid) + float2(dirRev));

         // don’t draw or pull over empty area
 
      }

//      for (i = 0; i < 4; ++i) {
//        canvas.write(middle[i], gid, i);
//        canvas.write(left[i], uint2(int2(gid) - int2(1,0)), i);
//        canvas.write(upperLeft[i], uint2(int2(gid) - int2(1,1)), i);
//        canvas.write(up[i], uint2(int2(gid) - int2(0,1)), i);
//        canvas.write(upperRight[i], uint2(int2(gid) - int2(-1,1)), i);
//        canvas.write(right[i], uint2(int2(gid) - int2(-1,0)), i);
//        canvas.write(lowerRight[i], uint2(int2(gid) - int2(-1,-1)), i);
//        canvas.write(down[i], uint2(int2(gid) - int2(0,-1)), i);
//        canvas.write(lowerLeft[i], uint2(int2(gid) - int2(1,-1)), i);
//      }

}


//constant bool hasLowerTexture [[function_constant(0)]];

kernel void drawDabs(constant Dab *dabArray [[ buffer(0) ]],
                    constant DabMeta *DabMeta [[ buffer(1) ]],
                    texture2d_array <half, access::read_write> activeLayer [[texture(0)]],
                    texture2d_array <half, access::read> smudgeBuckets [[texture(1)]],
                    texture2d_array <half, access::read> lowerLayer [[texture(2)]],
                    texture2d_array <half, access::sample> activeLayerSampler [[texture(3)]],
                    uint2 gid [[thread_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    
    // read the active layer pixels into dstX
    // we will modify this data repeatedly for each dab and write it back into the layer at the end
    // our data format is 16 channels. 12 log2 color channels (first 3 textures) and a metadata texture
    // metadata stores opacity, thickness, a thickness/opacity factor, and a "worked" factor
    half4 dstA = activeLayer.read(gid, 0);
    half4 dstB = activeLayer.read(gid, 1);
    half4 dstC = activeLayer.read(gid, 2);
    half4 dstMeta = activeLayer.read(gid, 3);
    
    // if there is a lower layer, read that in. We might use it
    half4 lowerA, lowerB, lowerC, lowerMeta;
    if (DabMeta->hasLowerTexture == 1) {
        lowerA = lowerLayer.read(gid, 0);
        lowerB = lowerLayer.read(gid, 1);
        lowerC = lowerLayer.read(gid, 2);
        lowerMeta = lowerLayer.read(gid, 3);
    }
    
    // for each dab, do a bunch of stuff and store it in the dst
    int dabCount = DabMeta->dabCount;
    for (int i=0; i < dabCount; i++) {
        // center of the dab to draw
        float2 center = (dabArray[i].pos);
        // translate the position using the global texOrigin coordinates
        half xOffset = half(gid.x) + DabMeta->texOrigin.x - center.x;
        half yOffset = half(gid.y) + DabMeta->texOrigin.y - center.y;
        
        // coverage is either on or off for a pixel and is a % chance basically
        // needs seeds to avoid obvious repeating noise
        if (dabArray[i].dabCoverage < 1.0) {
            half dabCoverageChance = rand(xOffset + tid, yOffset + tid, DabMeta->randomSeeds2.y);
            if ( dabCoverageChance > dabArray[i].dabCoverage ) continue;
        }
        
        // ratio of smudge bucket color to the brush color
        // 1.0 will smear existing paint
        half smudgeAmount = dabArray[i].smudgeAmount;
        half invSmudgeAmount = 1.0 - smudgeAmount;
        // opacity is opaqueness
        // Partially opaque and thick paint will be darker
        half opacity = clamp(half(dabArray[i].opacity),EPSILON, half(1.0));
        // volume AKA thickness. Scale to 10.0 units
        half volume = clamp(half(dabArray[i].volume * 10.0),EPSILON, half(10.0));
        // strength is ultimately the ratio of brush to existing canvas ratio, even
        // when erasing.  It's the strength of the effect of whatever you're doing.
        half strength = dabArray[i].strength;
        // radius of the dab to draw, in pixels
        half radius = dabArray[i].radius;
        // hardness is how much to feather the edge of a dab
        half hardness = dabArray[i].hardness;
        // eraser removes existing paint
        half eraser = 1.0 - dabArray[i].eraser;
        
        // jitter the opacity
        if (dabArray[i].opacityJitterChance > 0.0 && dabArray[i].opacityJitter > 0.0) {
            half opacityJitterChance = rand(xOffset + tid, yOffset + tid, DabMeta->randomSeeds.x);
            if ( opacityJitterChance < dabArray[i].opacityJitterChance) opacity *= max(half(1.0 - dabArray[i].opacityJitter), opacityJitterChance);
        }
        
        // jitter smudge
        if (dabArray[i].smudgeJitterChance > 0.0 && dabArray[i].smudgeJitter > 0.0) {
            half smudgeJitterChance = rand(yOffset + tid, xOffset + tid, DabMeta->randomSeeds.y);
            if (( smudgeJitterChance < dabArray[i].smudgeJitterChance )) smudgeAmount *= max(half(1.0 - dabArray[i].smudgeJitter), smudgeJitterChance);
        }

        // use a signed distance field to draw the shape
        half dist;
        // draw ellipse and/or squircle shape
        if ( dabArray[i].dabRatio < 1.0 || dabArray[i].dabShape > 0.0) {
            // major and minor axis lengths
            half a = pow((radius * half(dabArray[i].dabRatio)), half(2.0));
            half b = pow((radius), half(2.0));

            half rotation = dabArray[i].dabAngle;
            
            // apply affine transform for rotation and offset
            float x = xOffset * cos(rotation) + (yOffset) * sin(rotation);
            float y = -1 * ((xOffset) * sin(rotation)) + (yOffset) * cos(rotation);

            // standard equation for ellipse == 1 if point is exactly on the ellipse perimeter
            half distEllipse = (pow(x, 2.0) / a ) + (pow((y), (2.0)) / b);
            
            // squircle
            half n = dabArray[i].dabShapeMod;
            half distSquircle = max(half(0.0), half(pow(half(abs(x / (radius * dabArray[i].dabRatio))), n) + pow(half(abs(y / radius)), n)));

            // interpolate between ellipse and squircle shape
            dist = (1.0 - dabArray[i].dabShape) * distEllipse + dabArray[i].dabShape * distSquircle;
        } else {
            // optimization for just a simple circle
            dist = distance(float2(gid) + float2(DabMeta->texOrigin), center) / radius;
        }
        
        // if outside the ellipse, don't draw anything for this dab
        if (dist > 1.0 || dist < 0.0 || isnan(dist)) continue;
        // otherwise, use the distance to adjust strength to fade out w/ hardness parameter
        strength *= (1.0 - (pow(dist, half(30.0) * hardness)));
        half strengthInv = 1.0 - strength;
        
        // Smudge Bucket is just a small texture that stores
        // a bunch of colors that can be recalled on a per-dab
        // basis. Kind of good for matching bristles to their
        // own smudge color
        uint2 bucket = uint2(dabArray[i].smudgeBucket, 0);
        half4 smudgeBucketD = smudgeBuckets.read(bucket, 3);
        
        // This is weird and basically tries to uhh only paint when
        // the smudge bucket state is below or above the desired value
        // I don't even know why I made this
        half smudgeThicknessThreshold = dabArray[i].smudgeThicknessThreshold;
        if ((smudgeThicknessThreshold > 0.0 && smudgeBucketD.y < smudgeThicknessThreshold) || (smudgeThicknessThreshold < 0.0 && smudgeBucketD.y > 1.0 - -smudgeThicknessThreshold)) {
            continue;
        }
        half4 smudgeBucketA = smudgeBuckets.read(bucket, 0);
        half4 smudgeBucketB = smudgeBuckets.read(bucket, 1);
        half4 smudgeBucketC = smudgeBuckets.read(bucket, 2);
        
        
        // if there is a layer below the active layer, we might want
        // to interact with it somehow (read-only). Here we lift paint
        // up from the lower layer and mix it with the smudge color
        // but only if our paint is a solvent and can disolve it
        if (DabMeta->hasLowerTexture == 1) {
            // lowerMeta.y is volume/thickness
            half liftFac = lowerMeta.y * dabArray[i].solvent * 0.01;
            smudgeBucketA = smudgeBucketA * (1.0 - liftFac) + lowerA * liftFac;
            smudgeBucketB = smudgeBucketB * (1.0 - liftFac) + lowerB * liftFac;
            smudgeBucketC = smudgeBucketC * (1.0 - liftFac) + lowerC * liftFac;
            smudgeBucketD = smudgeBucketD * (1.0 - liftFac) + lowerMeta * liftFac;
        }
        
        // jitter the volume/thickness
        if (dabArray[i].volumeJitterChance > 0.0) {
            half volumeJitterChance = rand(yOffset + tid, xOffset + tid, DabMeta->randomSeeds2.x);
            if (( volumeJitterChance < dabArray[i].volumeJitterChance )) volume *= max(half(1.0 - dabArray[i].volumeJitter), volumeJitterChance);
        }
        
        // calculate volume/thickness
        half volumeTop = (( smudgeAmount * smudgeBucketD.y ) + (invSmudgeAmount * volume * eraser)) * strength;
        half volumeBottom =  strengthInv * dstMeta.y;
        half volumeResult = volumeTop + volumeBottom;
        
        // calculate opacity before applying thickness
        half topOpacity = (smudgeAmount * smudgeBucketD.x + invSmudgeAmount * opacity * eraser);
        half bottomOpacity = strengthInv * dstMeta.x;
        half opacityResult =  topOpacity * strength + bottomOpacity;
        
        // apply beer-lambert-like multiplier to the color based on opacity and thickness
        half beerMultiplier = (volume * ((half(1.0) - opacity))) + 1.0;
        
        
        half workedAmount = eraser * clamp(half(strength * dabArray[i].pressure + dstMeta.w), half(0.0), half(1000.0));
        half beerMultiplierTop = (smudgeAmount * smudgeBucketD.z + (1.0 - smudgeAmount) * beerMultiplier * eraser) * strength;
        half beerResult = beerMultiplierTop + strengthInv * dstMeta.z;
        
        // brush color, 12 channels, remember?
        // log2 encoded
        half4 color0 = half4(dabArray[i].color[0]);
        half4 color1 = half4(dabArray[i].color[1]);
        half4 color2 = half4(dabArray[i].color[2]);
        
        if ( dabArray[i].valueJitter != 0.0 && dabArray[i].valueJitterChance > 0.0) {
            half valueJitterChance = 1.0 - rand(xOffset + tid, yOffset + tid, DabMeta->randomSeeds.y);
            half valueJitter =  dabArray[i].valueJitter * valueJitterChance;
            
            if (valueJitterChance <= dabArray[i].valueJitterChance) {

                if (valueJitter < 0.0 ) {
                    valueJitter = -valueJitter;
                    for (uint i=0; i < 4; i++) {
                        color0[i] = max(EPSILON_LOG, color0[i] + valueJitter * EPSILON_LOG);
                        color1[i] = max(EPSILON_LOG, color1[i] + valueJitter * EPSILON_LOG);
                        color2[i] = max(EPSILON_LOG, color2[i] + valueJitter * EPSILON_LOG);
                       
                    }
                } else {
                    valueJitter = 1.0 - valueJitter;
                    for (uint i=0; i < 4; i++) {
                        color0[i] *= valueJitter;
                        color1[i] *= valueJitter;
                        color2[i] *= valueJitter;
                    }
                }
            }
        }
        
        if ( dabArray[i].colorJitter != 0.0 && dabArray[i].colorJitterChance > 0.0) {
            
            half colorJitterChance = rand(yOffset + tid, xOffset + tid, DabMeta->randomSeeds.w);
            half colorJitter = dabArray[i].colorJitter * colorJitterChance;
            
            if (colorJitterChance < dabArray[i].colorJitterChance) {
                if (colorJitter > 0.0) {
                    
                    half invColorJitter = 1.0 - colorJitter;
                    for (uint i=0; i < 3; i++) {
                        color0[i] = color0[i] * invColorJitter + colorJitter * color0[i + 1];
                        color1[i] = color1[i] * invColorJitter + colorJitter * color1[i + 1];
                        color2[i] = color2[i] * invColorJitter + colorJitter * color2[i + 1];
                    }
                    color0[3] = color0[3] * invColorJitter + colorJitter * color1[0];
                    color1[3] = color1[3] * invColorJitter + colorJitter * color2[0];
                    color2[3] = color2[3] * invColorJitter + colorJitter * color0[0];
                
                } else if (colorJitter < 0.0) {
                    colorJitter = -colorJitter;
                    half invColorJitter = 1.0 - colorJitter;
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

        // combine smudge, brush color,
        half4 colorA = (smudgeAmount * smudgeBucketA + (1.0 - smudgeAmount) * color0 * beerMultiplier * eraser) * strength;
        half4 colorB = (smudgeAmount * smudgeBucketB + (1.0 - smudgeAmount)  * color1 * beerMultiplier * eraser) * strength;
        half4 colorC = (smudgeAmount * smudgeBucketC + (1.0 - smudgeAmount)  * color2 * beerMultiplier * eraser) * strength;
       
        dstA = colorA + strengthInv * dstA;
        dstB =  colorB + strengthInv * dstB;
        dstC =  colorC + strengthInv * dstC;
        dstMeta = half4(clamp(opacityResult, half(0.0), half(1.0)), volumeResult, beerResult, workedAmount);
    }
    activeLayer.write(dstA, gid, 0);
    activeLayer.write(dstB, gid, 1);
    activeLayer.write(dstC, gid, 2);
    activeLayer.write(dstMeta, gid, 3);
}



kernel void spectralOver(texture2d_array<half, access::read> src [[texture(0)]],
                 texture2d_array<half, access::read_write> dst [[texture(1)]],
                 constant OverOp &overOp [[ buffer(0) ]],
                 uint2 gid [[thread_position_in_grid]]) {

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
    // max out volume at 20.0 so that things aren't bumpy forever w/ many layers
    half volume = clamp((srcThickness + dstThickness), half(0.0), half(10.0));

    
    half beerFac = (srcMeta.z * overOp.srcThickness + dstAndLayerOpacity * dstMeta.z);
    half workedAmount = (srcMeta.w * srcAndLayerOpacity + dstAndLayerOpacity + dstMeta.w);
    srcMeta = half4(clamp(opacity, half(0.0), half(1.0)), volume, beerFac, workedAmount);
    
    dst.write(srcS, gid, 0);
    dst.write(srcM, gid, 1);
    dst.write(srcL, gid, 2);
    dst.write(srcMeta, gid, 3);
}

// Vertex Function
vertex RasterizerData
vertexShader(uint vertexID [[ vertex_id ]],
             constant Vertex *vertexArray [[ buffer(0) ]],
             constant vector_uint2 *viewportSizePointer  [[ buffer(1) ]])

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
