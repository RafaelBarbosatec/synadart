import 'package:synadart/src/layers/layer.dart';
import 'package:synadart/src/networks/network.dart';
import 'package:synadart/src/networks/training/backpropagation.dart';

/// A `Network` model in which every `Layer` has one input and one output
/// tensor.
class Sequential extends Network with Backpropagation {
  final String _layersField = 'layers';
  final String _learningRateField = 'learningRate';

  /// Creates a `Sequential` model network.
  Sequential({
    required super.learningRate,
    super.layers,
  });

  /// Loads a model from a JSON .
  Sequential.fromMap(Map<String, dynamic> data) : super(learningRate: 0) {
    learningRate = data[_learningRateField];
    for (Map<String, dynamic> layer in data[_layersField]) {
      layers.add(Layer.fromJson(layer));
    }
  }

  /// Save the model to a JSON.
  Map<String, dynamic> toMap() {
    return {
      _layersField: layers.map((e) => e.toJson()).toList(),
      _learningRateField: learningRate,
    };
  }

  Sequential variation() {
    return Sequential(
      learningRate: learningRate,
    )..layers.addAll(layers.map((e) {
        return e.isInput ? e.copyWith() : e.variation();
      }).toList());
  }
}
