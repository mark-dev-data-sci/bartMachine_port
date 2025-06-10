"""
Test script for getting posterior samples from a BART model.

This script tests the bart_machine_get_posterior function in the zzz module,
which gets posterior samples from a BART model.
"""

import numpy as np
import pandas as pd
from bartmachine.zzz import (
    initialize_jvm,
    create_bart_machine,
    bart_machine_get_posterior,
    shutdown_jvm,
    convert_to_java_2d_array,
    get_posterior_samples
)

def debug_java_array(java_array):
    """
    Print debug information about a Java array.
    """
    print(f"Java array class: {java_array.getClass().getName()}")
    print(f"Java array length: {java_array.length}")
    
    # Try to get component type
    try:
        component_type = java_array.getClass().getComponentType()
        print(f"Component type: {component_type.getName()}")
        
        # Try to get component type of component type (for 2D arrays)
        try:
            component_type2 = component_type.getComponentType()
            print(f"Component type of component type: {component_type2.getName()}")
        except:
            print("Could not get component type of component type.")
    except:
        print("Could not get component type.")
    
    # Try to access first element
    try:
        first_element = java_array[0]
        print(f"First element class: {first_element.getClass().getName()}")
        print(f"First element: {first_element}")
        
        # If it's a 2D array, try to access first element of first element
        try:
            first_element_length = first_element.length
            print(f"First element length: {first_element_length}")
            
            if first_element_length > 0:
                first_element_first_element = first_element[0]
                print(f"First element of first element class: {first_element_first_element.getClass().getName()}")
                print(f"First element of first element: {first_element_first_element}")
        except:
            print("Could not access first element as array.")
    except:
        print("Could not access first element.")

def main():
    # Initialize JVM
    print("Initializing JVM...")
    initialize_jvm(debug=True)
    
    # Create sample data
    print("Creating sample data...")
    np.random.seed(42)
    X = np.random.randn(20, 5)
    y = np.sin(X[:, 0]) + 0.1 * np.random.randn(20)
    
    # Create BART machine
    print("Creating BART machine...")
    bart = create_bart_machine(
        X.tolist(), 
        y.tolist(), 
        num_trees=10, 
        num_burn_in=10, 
        num_iterations_after_burn_in=10
    )
    
    # Create test data
    print("Creating test data...")
    X_test = np.random.randn(5, 5)
    
    # Get posterior samples
    print("Getting posterior samples...")
    try:
        # Convert Python array to Java array
        X_test_java = convert_to_java_2d_array(X_test.tolist(), "double")
        
        # Call the Java method directly
        print("Calling getGibbsSamplesForPredictionPublic directly...")
        try:
            samples_java = bart.getGibbsSamplesForPredictionPublic(X_test_java, 1)
            print("Successfully called getGibbsSamplesForPredictionPublic.")
            print("Debug information for samples_java:")
            debug_java_array(samples_java)
            
            # Try to manually convert the samples
            print("\nTrying to manually convert the samples...")
            try:
                rows = samples_java.length
                result = []
                for i in range(rows):
                    row = samples_java[i]
                    cols = row.length
                    result.append([float(row[j]) for j in range(cols)])
                print(f"Successfully converted samples. Shape: {np.array(result).shape}")
                print(f"First few values: {np.array(result)[0, :5]}")
            except Exception as e:
                print(f"Error manually converting samples: {str(e)}")
        except Exception as e:
            print(f"Error calling getGibbsSamplesForPredictionPublic: {str(e)}")
        
        # Try the regular approach
        print("\nTrying the regular approach...")
        try:
            posterior = bart_machine_get_posterior(bart, X_test.tolist(), num_cores=1)
            print("Success! Got posterior samples.")
            print(f"Shape of posterior samples: {np.array(posterior['y_hat_posterior_samples']).shape}")
            print(f"Mean prediction: {np.mean(posterior['y_hat'])}")
        except Exception as e:
            print(f"Error with bart_machine_get_posterior: {str(e)}")
            
        # Try the get_posterior_samples function
        print("\nTrying the get_posterior_samples function...")
        try:
            samples = get_posterior_samples(bart, X_test.tolist(), num_cores=1)
            print("Success! Got posterior samples.")
            print(f"Shape of posterior samples: {np.array(samples).shape}")
        except Exception as e:
            print(f"Error with get_posterior_samples: {str(e)}")
    except Exception as e:
        print(f"Error getting posterior samples: {str(e)}")
    
    # Shutdown JVM
    print("Shutting down JVM...")
    shutdown_jvm()

if __name__ == "__main__":
    main()
