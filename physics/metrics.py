import tensorflow as tf

def true_stress(y_true, y_pred):
    """
    True Stress: Calculated during plastic deformation.
    It's based on the instantaneous cross-sectional area (different from the original area).
    """
    return tf.reduce_mean((y_pred - y_true) / y_true)

def elastic_modulus(y_true, features):
    """
    Elastic Modulus (Young's Modulus): Measure of stiffness of an elastic material.
    It's defined as the ratio of the stress along an axis over the strain along that axis in the range of elastic deformation.
    """
    # Extracting the feature 'Standardkraft (MPa)' which represents stress
    stress = features[:, 0]  # Adjust index based on your features ordering

    # Calculating Elastic Modulus
    strain = y_true  # Strain is the target variable
    return tf.reduce_mean(stress / strain)

def yield_strength(y_true, y_pred):
    """
    Yield Strength: The stress at which a material begins to deform plastically.
    Prior to the yield point, the material will deform elastically and will return to its original shape when the applied stress is removed.
    """
    # The yield strength could be approximated by the stress value at which plastic deformation begins.
    # This function assumes that the 'y_true' and 'y_pred' values represent stress values.
    return tf.reduce_max(y_true)