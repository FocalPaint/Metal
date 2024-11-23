# Metal
Metal Shader Code reference

# Custom Brush Shaders

Download the Beta: https://testflight.apple.com/join/1adnuUHv

Yes, that's right. Now you can create your own custom Compute Functions in Metal Shading Language:

https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf

Quick video demo: https://www.youtube.com/watch?v=kiZCT-uVaaA

Everything in the Metal Standard Library should be available. If something doesn't work, please file an issue.
Typos and compiler errors should appear in a (rather small) pop up.

To make things easier, you can download the FocalPaintMetal project and edit your shader in Xcode:

There's a custom shader function `customBrushShader` already defined here: 
https://github.com/FocalPaint/Metal/blob/main/Shaders.metal

so you can just edit it and copy and paste into FocalPaint by following these instructions:

To create a custom shader, just open the brush editor (tap a brush in the brush menu) and click the `</>` button:

![image](https://user-images.githubusercontent.com/6015639/164361195-9d5daac8-370a-456e-a50c-08d3e0b16883.png)

You'll see black text box appear where you can type.  Click the `</>`  again to close the editor and compile the function(s):

![image](https://user-images.githubusercontent.com/6015639/164362256-636515d6-179c-4f3a-ae50-0d5515f4b03a.png)


You can use the onscreen keyboard to type out a function, or hopefully you can use [Universal Control](https://support.apple.com/en-us/HT212757) on a nearby Mac to make things easier.

Here's a really simple example that just fills the buffers (defined by the dab sizes you draw) with a solid color. Your custom shader must be named `customBrushShader` and must
have the same arguments as this example, but you can also include other functions, structs, etc.  Continue reading about the Dab and DabMeta Types, which you'll need to understand at least a tiny bit about.

# Simple Example

```
// this just fills the whole buffer with the brush color, skipping all other settings, etc

kernel void customBrushShader(
  constant Dab *dabArray [[ buffer(0) ]],
  constant DabMeta *dabMeta [[ buffer(1) ]],
  texture2d_array <half, access::read_write> canvas [[texture(0)]],
  texture2d_array <half, access::read> smudge [[texture(1)]],
  texture2d_array <half, access::read> lowerCanvas [[texture(2)]], // optional
  uint2 gid [[thread_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]]) 
  {

    half4 color0 = half4(dabArray[0].color[0]);
    half4 color1 = half4(dabArray[0].color[1]);
    half4 color2 = half4(dabArray[0].color[2]);

    canvas.write(color0, gid, 0);
    canvas.write(color1, gid, 1);
    canvas.write(color2, gid, 2);
    canvas.write(half4(1.0, 10.0,  1.0, 1.0), gid, 3);
}
```


You'll notice the first argument is an Array of `Dab`s, and the second argument is a single `DabMeta` object. Assuming you want to draw dabs, you may want to loop over the DabArray using the `DabMeta.dabCount` property.  Here are their definitions, these are included for you so you should NOT copy these into your shader code:

```
// a circle-like blob color or effect (eraser, etc) to apply to a canvas
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
    
    // how strong to pick up color from layer below active layer
    // only when smudging
    float solvent;
};


// information that is applicable for all the dabs
// in the dabarray
struct DabMeta {
    // how many dabs to draw in this draw call
    // should loop over dabArray, or not; do whatever you want
    uint32_t dabCount;
    
    // global origin coordinates of texture
    // the texture we have in the shader is likely much smaller
    // than the whole canvas, so this is the origin of that texture
    // in the context of the whole canvas.  (10,200) would mean the start of
    // this texture (upper left corner) is 10 pixels right and 200 pixels down 
    // from the upper-left corner of the whole canvas
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
```

You may notice that a `Dab` does not have all the same settings that are available in the brush editor. That is because I hadn't thought of any uses for those setting in the shader yet, so I didn't add them. A lot of them wouldn't really make sense, either. Settings like (regular) Jitter affect the coordinates of where the Dabs are placed.  The shader texture size itself is determined by the extents of the Dabs in the Dab Array, so it doesn't make much sense to try to use Jitter in the shader, since you likely can't move the dab much without moving it outside the texture boundary.  That said, it might still be interesting to have all the settings (and inputs as well) available in the shader  The only input right now is Pressure.

# Default Brush Shader

Here is the default brush shader, which you can draw inspiration from, laugh at, copy, or edit and put into your brush shader code editor:

```

constant half EPSILON = 0.0002;
constant half EPSILON_LOG = -12.287712379549449;

// https://developer.apple.com/library/archive/samplecode/MetalShaderShowcase/Listings/MetalShaderShowcase_AAPLWoodShader_metal.html
// Generate a random float in the range [0.0f, 1.0f] using x, y, and z (based on the xor128 algorithm)
float rand(int x, int y, int z)
{
    int seed = x + y * 57 + z * 241;
    seed= (seed<< 13) ^ seed;
    return (( 1.0 - ( (seed * (seed * seed * 15731 + 789221) + 1376312589) & 2147483647) / 1073741824.0f) + 1.0f) / 2.0f;
}

kernel void customBrushShader(constant Dab *dabArray [[ buffer(0) ]],
                    constant DabMeta *DabMeta [[ buffer(1) ]],
                    texture2d_array <half, access::read_write> activeLayer [[texture(0)]],
                    texture2d_array <half, access::read> smudgeBuckets [[texture(1)]],
                    texture2d_array <half, access::read> lowerLayer [[texture(2)]],
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
```

# What's up with the Texture Arrays?

Focal Paint uses 4 textures for each layer: 12 spectral color channels (log2 encoded) and 4 metadata channels (opacity, thickness, a Beer-Lambert factor, and a "worked" factor).  You can set the Beer-Lambert factor and "worked" factors to 1.0 if you don't want to use them.  Colors are normalized by dividing by the Beer-Lambert factor when using the color picker.  The "worked" factor affects how the background canvas texture is applied; the lower the "worked" about, the less the paint gets into the "crevices" of the texture.  That's the idea, anyway.

# Future

If this works out well, it might be neat to add ways to have custom shaders for other parts of the pipeline; custom layer/compositing shaders and post-compositing shaders (such as bump mapping).  For now those remain static.
