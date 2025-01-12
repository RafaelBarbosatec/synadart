// ignore_for_file: public_member_api_docs, sort_constructors_first
import 'dart:io';
import 'dart:math';

import 'package:sprint/sprint.dart';
import 'package:synadart/src/activation.dart';
import 'package:synadart/src/utils/mathematical_operations.dart';
import 'package:synadart/src/utils/utils.dart';
import 'package:synadart/src/utils/value_generator.dart';

/// A representation of a single `Neuron` within a `Network` of `Layers` (of
/// `Neurons`).  A `Neuron` can take on multiple forms and functions, the most
/// basic of which being the taking of the weighted sum of [inputs] and
/// [weights], and passing it on to the next `Neuron`.
class Neuron {
  /// `Sprint` instance for logging messages.
  final Sprint log = Sprint('Neuron');

  /// The activation algorithm used for determining this `Neuron`'s level of
  /// activation.
  late final ActivationFunction activation;

  /// The derivative of the activation algorithm.
  late final ActivationFunction activationPrime;

  /// The algorithm used for this `Neuron`
  late final ActivationAlgorithm activationAlgorithm;

  /// The sensitivity of this `Neuron` to the function adjusting [weights]
  /// during training.
  ///
  ///
  final double learningRate;

  /// The weights of connections to the precedent `Neurons`, which can be
  /// imagined as how influential each `Neuron` in the preceding layer is on
  /// this `Neuron`'s [output].
  List<double> weights = [];

  /// The inputs received from the precedent `Layer`, combined with [weights] or
  /// connections to `Neurons` from the previous `Layer` in order to create an
  /// [output].
  List<double> inputs = [];

  /// Specifies whether this `Neuron` belongs to an input `Layer`. This is used
  /// to determine the source of output, because a parentless `Neuron` will not
  /// have any connections.
  bool isInput = false;

  /// Keys used to identify this `Neuron` once parsed to [Map].
  static const _weightsField = 'weights';
  static const _activationField = 'activation';
  static const _learningRateField = 'learningRate';

  /// Creates a `Neuron` with the specified `ActivationAlgorithm`, which is then
  /// resolved to an `ActivationFunction`.
  ///
  /// [activationAlgorithm] - The algorithm used for 'activating' this `Neuron`,
  /// i.e. determining how 'active' this `Neuron` is by shrinking the weighted
  /// sum of this `Neuron`'s [weights] and [inputs] to a more appropriate range,
  /// such as 0–1.
  ///
  /// [parentLayerSize] - The number of connections this `Neuron` has, i.e. how
  /// many `Neurons` from the previous layer this `Neuron` is connected to.
  ///
  /// [learningRate] - A value between 0 (exclusive) and 1 (inclusive) that
  /// indicates how sensitive this `Neuron` is to [weights] adjustments.
  ///
  /// [weights] - (Optional) Weights of connections to `Neuron`s in the previous
  /// `Layer`.  If the [weights] aren't provided, they will be generated
  /// randomly.
  Neuron({
    required this.activationAlgorithm,
    required int parentLayerSize,
    required this.learningRate,
    List<double> weights = const [],
  }) {
    activation = resolveActivationAlgorithm(activationAlgorithm);
    activationPrime = resolveActivationDerivative(activationAlgorithm);

    if (parentLayerSize == 0) {
      isInput = true;
      this.weights.add(1);
      return;
    }

    if (weights.isEmpty) {
      // Calculate the range extreme based on the number of parent neurons.
      final limit = 1 / sqrt(parentLayerSize);
      // Generate random values for the connection weights.
      this.weights.addAll(
            generateListWithRandomDoubles(
              size: parentLayerSize,
              from: -limit,
              to: limit,
            ),
          );
      return;
    }

    if (weights.length != parentLayerSize) {
      log.severe(
        'The number of weights supplied to this neuron does not match the '
        'number of connections to neurons in the parent layer.',
      );
      exit(0);
    }

    // ignore: prefer_initializing_formals
    this.weights = weights;
    return;
  }

  /// Accepts a single [input] or multiple [inputs] by assigning them to the
  /// [inputs] of this `Neuron`.
  void accept({List<double>? inputs, double? input}) {
    if (inputs == null && input == null) {
      log.severe('Attempted to accept without any inputs.');
      exit(0);
    }

    if (!isInput && inputs != null) {
      this.inputs = inputs;
      return;
    }

    if (this.inputs.isNotEmpty) {
      this.inputs.first = input!;
    } else {
      this.inputs.add(input!);
    }
  }

  /// Adjusts this `Neuron`'s [weights] and returns their adjusted values.
  List<double> adjust({required double weightMargin}) {
    final adjustedWeights = <double>[];

    for (var index = 0; index < weights.length; index++) {
      adjustedWeights.add(weightMargin * weights[index]);
      weights[index] -= learningRate *
          -weightMargin *
          activationPrime(() => dot(inputs, weights)) * // δa/δz
          inputs[index]; // δz/δw
    }

    return adjustedWeights;
  }

  /// If this `Neuron` is an 'input' `Neuron`, it will output its sole input
  /// because it has no parent neurons and therefore has no weights.  Otherwise,
  /// it will output the weighted sum of the [inputs] and [weights], passed
  /// through the activation function.
  double get output =>
      weights.isEmpty ? inputs.first : activation(() => dot(inputs, weights));

  /// Create a `Neuron` from the it's JSON Model
  factory Neuron.fromJson(Map<String, dynamic> json) {
    final activationIndex = json[_activationField] as int;
    final weights = List<double>.from(json[_weightsField] as List);
    return Neuron(
      activationAlgorithm: ActivationAlgorithm.values[activationIndex],
      learningRate: json[_learningRateField] as double,
      weights: weights,
      parentLayerSize: weights.length,
    );
  }

  /// Parse this `Neuron` to a JSON Model
  Map<String, dynamic> toJson() => <String, dynamic>{
        _weightsField: weights,
        _activationField: activationAlgorithm.index,
        _learningRateField: learningRate,
      };

  Neuron copyWith({
    ActivationFunction? activation,
    ActivationFunction? activationPrime,
    ActivationAlgorithm? activationAlgorithm,
    double? learningRate,
    List<double>? weights,
  }) {
    return Neuron(
      activationAlgorithm: activationAlgorithm ?? this.activationAlgorithm,
      learningRate: learningRate ?? this.learningRate,
      weights: weights ?? this.weights,
      parentLayerSize: (weights ?? this.weights).length,
    );
  }

  Neuron variation({Mutation? mutation}) {
    var random = Random();
    final limit = 1 / sqrt(weights.length);
    return copyWith(
      weights: weights
          .map(mutation ??
              (e) {
                switch (random.nextInt(4)) {
                  case 0:
                    return nextDouble(from: -limit, to: limit);
                  case 1:
                    return e + random.nextDouble();
                  case 2:
                    return e * random.nextDouble();
                  default:
                    return e;
                }
              })
          .toList(),
    );
  }
}
