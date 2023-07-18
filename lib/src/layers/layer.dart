// ignore_for_file: public_member_api_docs, sort_constructors_first
import 'dart:io';

import 'package:sprint/sprint.dart';
import 'package:synadart/src/activation.dart';
import 'package:synadart/src/neurons/neuron.dart';
import 'package:synadart/src/utils/mathematical_operations.dart';

/// Representation of a single `Layer` inside a `Network`, more accurately a
/// 'column' of `Neurons` that can be manipulated through accepting new data and
/// trained.
class Layer {
  static const String _activationField = 'activation';
  static const String _neuronsField = 'neurons';
  static const String _isInputField = 'isInput';

  /// `Sprint` instance for logging messages.
  final Sprint log = Sprint('Layer');

  /// The algorithm used for activating `Neurons`.
  final ActivationAlgorithm activation;

  /// The `Neurons` part of this `Layer`.
  final List<Neuron> neurons;

  /// The number of `Neurons` this `Layer` comprises.
  final int size;

  /// Specifies whether or not this `Layer` is an input `Layer`.  This is used
  /// to determine how inputs should be accepted by each neuron in this `Layer`.
  bool isInput = false;

  /// Creates a `Layer` with the specified activation algorithm that is then
  /// passed to and resolved by `Neurons`.
  ///
  /// [size] - The number of `Neurons` this `Layer` is to house.
  ///
  /// [activation] - The algorithm used for determining how active `Neurons` are
  /// contained within this layer.
  Layer({
    required this.size,
    required this.activation,
    List<Neuron>? neurons,
  }) : neurons = neurons ?? [] {
    if (size < 1) {
      log.severe('A layer must contain at least one neuron.');
      exit(0);
    }
  }

  /// Initialises this `Layer` using the parameters passed into it by the
  /// `Network` in which the `Layer` is housed.
  ///
  /// [parentLayerSize] - The number of 'connections' this `Layer` is in
  /// disposition of.  In other words, the number of `Neurons` the previous
  /// `Layer` houses.  This number be equal to zero if this `Layer` is an input
  /// `Layer`.
  ///
  /// [learningRate] - A value between 0 (exclusive) and 1 (inclusive),
  /// indicating how sensitive the `Neurons` within this `Layer` are to
  /// adjustments of their weights.
  void initialise({
    required int parentLayerSize,
    required double learningRate,
  }) {
    isInput = parentLayerSize == 0;

    neurons.addAll(
      Iterable.generate(
        size,
        (_) => Neuron(
          activationAlgorithm: activation,
          parentLayerSize: parentLayerSize,
          learningRate: learningRate,
        ),
      ),
    );
  }

  /// Accept a single input or multiple [inputs] by assigning them sequentially
  /// to the inputs of the `Neurons` housed within this `Layer`.
  ///
  /// If [isInput] is equal to true, each `Neuron` within this `Layer` will only
  /// accept a single input corresponding to its index within the [neurons]
  /// list.
  void accept(List<double> inputs) {
    if (isInput) {
      for (var index = 0; index < neurons.length; index++) {
        neurons[index].accept(input: inputs[index]);
      }
      return;
    }

    for (final neuron in neurons) {
      neuron.accept(inputs: inputs);
    }
  }

  /// Adjusts weights of each `Neuron` based on its respective weight margin,
  /// and returns the new [weightMargins] for the previous `Layer` (We are
  /// moving backwards during propagation).
  List<double> propagate(List<double> weightMargins) {
    final newWeightMargins = <List<double>>[];

    for (final neuron in neurons) {
      newWeightMargins
          .add(neuron.adjust(weightMargin: weightMargins.removeAt(0)));
    }

    return newWeightMargins.reduce(add);
  }

  /// Returns a list of this `Layer`'s `Neuron`s' outputs
  List<double> get output =>
      List<double>.from(neurons.map<double>((neuron) => neuron.output));

  factory Layer.fromJson(Map<String, dynamic> json) {
    final activation =
        ActivationAlgorithm.values[json[_activationField] as int];
    final neurons = (json[_neuronsField] as List<Map<String, dynamic>>)
        .map(Neuron.fromJson)
        .toList();
    final isInput = json[_isInputField];

    return Layer(
      size: neurons.length,
      activation: activation,
      neurons: neurons,
    )
      // Only the firt layer is an input layer
      ..isInput = isInput;
  }

  Map<String, dynamic> toJson() {
    return {
      _activationField: activation.index,
      _neuronsField: neurons.map((e) => e.toJson()).toList(),
      _isInputField: isInput,
    };
  }

  Layer variation({double variationRate = 0.1}) {
    return copyWith(
      neurons: neurons
          .map((e) => e.variation(variationRate: variationRate))
          .toList(),
    );
  }

  Layer copyWith({
    ActivationAlgorithm? activation,
    bool? isInput,
    List<Neuron>? neurons,
  }) {
    return Layer(
      activation: activation ?? this.activation,
      size: (neurons ?? this.neurons).length,
      neurons: neurons ?? this.neurons,
    )..isInput = isInput ?? this.isInput;
  }
}
