function [lgraph] = model_dccnlsc(input_size,output_size)
lgraph = layerGraph();

tempLayers = [
    imageInputLayer(input_size,"Name","input_1","Normalization","zscore")
    convolution2dLayer([7 7],64,"Name","conv1|conv","BiasLearnRateFactor",0,"Padding",[3 3 3 3],"Stride",[2 2])
    batchNormalizationLayer("Name","conv1|bn")
    reluLayer("Name","conv1|relu")
    averagePooling2dLayer([2 2],"Name","pool2_pool","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv3_block1_0_bn")
    reluLayer("Name","conv3_block1_0_relu")
    convolution2dLayer([1 1],128,"Name","conv3_block1_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv3_block1_1_bn")
    reluLayer("Name","conv3_block1_1_relu")
    convolution2dLayer([3 3],32,"Name","conv3_block1_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv3_block1_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv3_block2_0_bn")
    reluLayer("Name","conv3_block2_0_relu")
    convolution2dLayer([1 1],128,"Name","conv3_block2_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv3_block2_1_bn")
    reluLayer("Name","conv3_block2_1_relu")
    convolution2dLayer([3 3],32,"Name","conv3_block2_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv3_block2_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv3_block3_0_bn")
    reluLayer("Name","conv3_block3_0_relu")
    convolution2dLayer([1 1],128,"Name","conv3_block3_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv3_block3_1_bn")
    reluLayer("Name","conv3_block3_1_relu")
    convolution2dLayer([3 3],32,"Name","conv3_block3_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv3_block3_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv3_block4_0_bn")
    reluLayer("Name","conv3_block4_0_relu")
    convolution2dLayer([1 1],128,"Name","conv3_block4_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv3_block4_1_bn")
    reluLayer("Name","conv3_block4_1_relu")
    convolution2dLayer([3 3],32,"Name","conv3_block4_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv3_block4_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv3_block5_0_bn")
    reluLayer("Name","conv3_block5_0_relu")
    convolution2dLayer([1 1],128,"Name","conv3_block5_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv3_block5_1_bn")
    reluLayer("Name","conv3_block5_1_relu")
    convolution2dLayer([3 3],32,"Name","conv3_block5_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv3_block5_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv3_block6_0_bn")
    reluLayer("Name","conv3_block6_0_relu")
    convolution2dLayer([1 1],128,"Name","conv3_block6_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv3_block6_1_bn")
    reluLayer("Name","conv3_block6_1_relu")
    convolution2dLayer([3 3],32,"Name","conv3_block6_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv3_block6_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv3_block7_0_bn")
    reluLayer("Name","conv3_block7_0_relu")
    convolution2dLayer([1 1],128,"Name","conv3_block7_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv3_block7_1_bn")
    reluLayer("Name","conv3_block7_1_relu")
    convolution2dLayer([3 3],32,"Name","conv3_block7_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv3_block7_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv3_block8_0_bn")
    reluLayer("Name","conv3_block8_0_relu")
    convolution2dLayer([1 1],128,"Name","conv3_block8_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv3_block8_1_bn")
    reluLayer("Name","conv3_block8_1_relu")
    convolution2dLayer([3 3],32,"Name","conv3_block8_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv3_block8_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv3_block9_0_bn")
    reluLayer("Name","conv3_block9_0_relu")
    convolution2dLayer([1 1],128,"Name","conv3_block9_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv3_block9_1_bn")
    reluLayer("Name","conv3_block9_1_relu")
    convolution2dLayer([3 3],32,"Name","conv3_block9_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv3_block9_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv3_block10_0_bn")
    reluLayer("Name","conv3_block10_0_relu")
    convolution2dLayer([1 1],128,"Name","conv3_block10_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv3_block10_1_bn")
    reluLayer("Name","conv3_block10_1_relu")
    convolution2dLayer([3 3],32,"Name","conv3_block10_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv3_block10_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv3_block11_0_bn")
    reluLayer("Name","conv3_block11_0_relu")
    convolution2dLayer([1 1],128,"Name","conv3_block11_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv3_block11_1_bn")
    reluLayer("Name","conv3_block11_1_relu")
    convolution2dLayer([3 3],32,"Name","conv3_block11_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv3_block11_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv3_block12_0_bn")
    reluLayer("Name","conv3_block12_0_relu")
    convolution2dLayer([1 1],128,"Name","conv3_block12_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv3_block12_1_bn")
    reluLayer("Name","conv3_block12_1_relu")
    convolution2dLayer([3 3],32,"Name","conv3_block12_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","conv3_block12_concat")
    batchNormalizationLayer("Name","pool3_bn")
    reluLayer("Name","pool3_relu")
    convolution2dLayer([1 1],256,"Name","pool3_conv","BiasLearnRateFactor",0)
    averagePooling2dLayer([2 2],"Name","pool3_pool","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block1_0_bn")
    reluLayer("Name","conv4_block1_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block1_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block1_1_bn")
    reluLayer("Name","conv4_block1_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block1_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")
    convolution2dLayer([3 3],1,"Name","conv_3","Padding","same")
    averagePooling2dLayer([14 14],"Name","avgpool2d_3","Padding","same")
    fullyConnectedLayer(200,"Name","fc_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu")
    convolution2dLayer([3 3],32,"Name","conv","Padding","same")
    averagePooling2dLayer([5 5],"Name","avgpool2d","Padding","same")
    fullyConnectedLayer(10,"Name","fc")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block1_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block2_0_bn")
    reluLayer("Name","conv4_block2_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block2_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block2_1_bn")
    reluLayer("Name","conv4_block2_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block2_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block2_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block3_0_bn")
    reluLayer("Name","conv4_block3_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block3_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block3_1_bn")
    reluLayer("Name","conv4_block3_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block3_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block3_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block4_0_bn")
    reluLayer("Name","conv4_block4_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block4_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block4_1_bn")
    reluLayer("Name","conv4_block4_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block4_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block4_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block5_0_bn")
    reluLayer("Name","conv4_block5_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block5_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block5_1_bn")
    reluLayer("Name","conv4_block5_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block5_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block5_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block6_0_bn")
    reluLayer("Name","conv4_block6_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block6_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block6_1_bn")
    reluLayer("Name","conv4_block6_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block6_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv4_block6_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv4_block7_0_bn")
    reluLayer("Name","conv4_block7_0_relu")
    convolution2dLayer([1 1],128,"Name","conv4_block7_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv4_block7_1_bn")
    reluLayer("Name","conv4_block7_1_relu")
    convolution2dLayer([3 3],32,"Name","conv4_block7_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_1")
    batchNormalizationLayer("Name","bn")
    globalAveragePooling2dLayer("Name","avg_pool")
    fullyConnectedLayer(200,"Name","fc_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(3,"Name","depthcat_2")
    fullyConnectedLayer(output_size,"Name","new_fc","BiasLearnRateFactor",10,"WeightLearnRateFactor",10)
    softmaxLayer("Name","fc1000_softmax")
    classificationLayer("Name","new_classoutput")];
lgraph = addLayers(lgraph,tempLayers);

lgraph = connectLayers(lgraph,"pool2_pool","conv3_block1_0_bn");
lgraph = connectLayers(lgraph,"pool2_pool","conv3_block1_concat/in1");
lgraph = connectLayers(lgraph,"conv3_block1_2_conv","conv3_block1_concat/in2");
lgraph = connectLayers(lgraph,"conv3_block1_concat","conv3_block2_0_bn");
lgraph = connectLayers(lgraph,"conv3_block1_concat","conv3_block2_concat/in1");
lgraph = connectLayers(lgraph,"conv3_block2_2_conv","conv3_block2_concat/in2");
lgraph = connectLayers(lgraph,"conv3_block2_concat","conv3_block3_0_bn");
lgraph = connectLayers(lgraph,"conv3_block2_concat","conv3_block3_concat/in1");
lgraph = connectLayers(lgraph,"conv3_block3_2_conv","conv3_block3_concat/in2");
lgraph = connectLayers(lgraph,"conv3_block3_concat","conv3_block4_0_bn");
lgraph = connectLayers(lgraph,"conv3_block3_concat","conv3_block4_concat/in1");
lgraph = connectLayers(lgraph,"conv3_block4_2_conv","conv3_block4_concat/in2");
lgraph = connectLayers(lgraph,"conv3_block4_concat","conv3_block5_0_bn");
lgraph = connectLayers(lgraph,"conv3_block4_concat","conv3_block5_concat/in1");
lgraph = connectLayers(lgraph,"conv3_block5_2_conv","conv3_block5_concat/in2");
lgraph = connectLayers(lgraph,"conv3_block5_concat","conv3_block6_0_bn");
lgraph = connectLayers(lgraph,"conv3_block5_concat","conv3_block6_concat/in1");
lgraph = connectLayers(lgraph,"conv3_block6_2_conv","conv3_block6_concat/in2");
lgraph = connectLayers(lgraph,"conv3_block6_concat","conv3_block7_0_bn");
lgraph = connectLayers(lgraph,"conv3_block6_concat","conv3_block7_concat/in1");
lgraph = connectLayers(lgraph,"conv3_block7_2_conv","conv3_block7_concat/in2");
lgraph = connectLayers(lgraph,"conv3_block7_concat","conv3_block8_0_bn");
lgraph = connectLayers(lgraph,"conv3_block7_concat","conv3_block8_concat/in1");
lgraph = connectLayers(lgraph,"conv3_block8_2_conv","conv3_block8_concat/in2");
lgraph = connectLayers(lgraph,"conv3_block8_concat","conv3_block9_0_bn");
lgraph = connectLayers(lgraph,"conv3_block8_concat","conv3_block9_concat/in1");
lgraph = connectLayers(lgraph,"conv3_block9_2_conv","conv3_block9_concat/in2");
lgraph = connectLayers(lgraph,"conv3_block9_concat","conv3_block10_0_bn");
lgraph = connectLayers(lgraph,"conv3_block9_concat","conv3_block10_concat/in1");
lgraph = connectLayers(lgraph,"conv3_block10_2_conv","conv3_block10_concat/in2");
lgraph = connectLayers(lgraph,"conv3_block10_concat","conv3_block11_0_bn");
lgraph = connectLayers(lgraph,"conv3_block10_concat","conv3_block11_concat/in1");
lgraph = connectLayers(lgraph,"conv3_block11_2_conv","conv3_block11_concat/in2");
lgraph = connectLayers(lgraph,"conv3_block11_concat","conv3_block12_0_bn");
lgraph = connectLayers(lgraph,"conv3_block11_concat","conv3_block12_concat/in1");
lgraph = connectLayers(lgraph,"conv3_block12_2_conv","conv3_block12_concat/in2");
lgraph = connectLayers(lgraph,"pool3_pool","conv4_block1_0_bn");
lgraph = connectLayers(lgraph,"pool3_pool","batchnorm_3");
lgraph = connectLayers(lgraph,"pool3_pool","batchnorm");
lgraph = connectLayers(lgraph,"pool3_pool","conv4_block1_concat/in1");
lgraph = connectLayers(lgraph,"fc_3","depthcat_2/in2");
lgraph = connectLayers(lgraph,"conv4_block1_2_conv","conv4_block1_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block1_concat","conv4_block2_0_bn");
lgraph = connectLayers(lgraph,"conv4_block1_concat","conv4_block2_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block2_2_conv","conv4_block2_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block2_concat","conv4_block3_0_bn");
lgraph = connectLayers(lgraph,"conv4_block2_concat","conv4_block3_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block3_2_conv","conv4_block3_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block3_concat","conv4_block4_0_bn");
lgraph = connectLayers(lgraph,"conv4_block3_concat","conv4_block4_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block4_2_conv","conv4_block4_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block4_concat","conv4_block5_0_bn");
lgraph = connectLayers(lgraph,"conv4_block4_concat","conv4_block5_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block5_2_conv","conv4_block5_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block5_concat","conv4_block6_0_bn");
lgraph = connectLayers(lgraph,"conv4_block5_concat","conv4_block6_concat/in1");
lgraph = connectLayers(lgraph,"conv4_block6_2_conv","conv4_block6_concat/in2");
lgraph = connectLayers(lgraph,"conv4_block6_concat","conv4_block7_0_bn");
lgraph = connectLayers(lgraph,"conv4_block6_concat","depthcat_1/in2");
lgraph = connectLayers(lgraph,"conv4_block7_2_conv","depthcat_1/in1");
lgraph = connectLayers(lgraph,"fc_4","depthcat_2/in1");
lgraph = connectLayers(lgraph,"fc","depthcat_2/in3");

clear tempLayers;

plot(lgraph);
end