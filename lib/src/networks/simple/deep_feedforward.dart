import 'dart:io';

import 'package:synadart/src/activation.dart';/*  */
import 'package:synadart/src/networks/network.dart';
import 'package:synadart/src/networks/training/backpropagation.dart';

/// A more complex feed-forward `Network` model with one or more hidden `Layer`s
class DeepFeedForward extends Network with Backpropagation {
  /// Constructs a deep feed-forward `Network`
  DeepFeedForward({
    required ActivationAlgorithm activationAlgorithm,
    required List<int> layerSizes,
    required double learningRate,
  }) : super(
    activationAlgorithm: activationAlgorithm, 
    layerSizes: layerSizes,
    learningRate: learningRate,
  ) {
    if (layerSizes.length < 3) {
      log.error('A deep feed-forward network model must have at least one hidden layer.');
      exit(0);
    }
  }
}