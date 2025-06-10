"""
Simple script to check if we can access Java classes.
"""

import os
import sys
from py4j.java_gateway import JavaGateway, GatewayParameters, launch_gateway

def main():
    # Find JAR files
    jar_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "java")
    jar_files = [os.path.join(jar_dir, f) for f in os.listdir(jar_dir) if f.endswith('.jar')]
    
    print(f"Found JAR files: {jar_files}")
    
    # Set up the classpath with all JAR files
    classpath = os.pathsep.join(jar_files)
    
    # Launch the JVM
    print("Launching JVM...")
    port = launch_gateway(
        classpath=classpath,
        die_on_exit=True,
        javaopts=["-Xmx1024m"]
    )
    
    # Connect to the gateway
    print(f"Connecting to gateway on port {port}...")
    gateway = JavaGateway(
        gateway_parameters=GatewayParameters(port=port)
    )
    
    # Check Java version
    java_version = gateway.jvm.java.lang.System.getProperty("java.runtime.version")
    print(f"Java version: {java_version}")
    
    # Check if bartMachine package is available
    print("\nChecking if bartMachine package is available...")
    try:
        bart_machine_package = gateway.jvm.bartMachine
        print("bartMachine package is available.")
        
        # List all classes in the bartMachine package
        print("\nClasses in bartMachine package:")
        for item in dir(bart_machine_package):
            if not item.startswith("_"):
                print(f"- {item}")
        
        # Check if BartMachineWrapper class is available
        print("\nChecking if BartMachineWrapper class is available...")
        try:
            wrapper_class = gateway.jvm.bartMachine.BartMachineWrapper
            print(f"BartMachineWrapper class is available: {wrapper_class}")
            
            # Try to instantiate the class directly
            print("\nTrying to instantiate BartMachineWrapper class directly...")
            try:
                wrapper_instance = wrapper_class()
                print("Successfully instantiated BartMachineWrapper class.")
                print(f"Instance: {wrapper_instance}")
                
                # Try to call a method on the wrapper instance
                print("\nTrying to call a method on the wrapper instance...")
                try:
                    # Call the numSamplesAfterBurning method
                    num_samples = wrapper_instance.numSamplesAfterBurning()
                    print(f"Number of samples after burning: {num_samples}")
                except Exception as e:
                    print(f"Failed to call method on wrapper instance: {str(e)}")
            except Exception as e:
                print(f"Failed to instantiate BartMachineWrapper class directly: {str(e)}")
                
            # Try to instantiate the bartMachineRegressionMultThread class directly
            print("\nTrying to instantiate bartMachineRegressionMultThread class directly...")
            try:
                regression_class = gateway.jvm.bartMachine.bartMachineRegressionMultThread
                regression_instance = regression_class()
                print("Successfully instantiated bartMachineRegressionMultThread class.")
                print(f"Instance: {regression_instance}")
            except Exception as e:
                print(f"Failed to instantiate bartMachineRegressionMultThread class: {str(e)}")
        except Exception as e:
            print(f"BartMachineWrapper class is not available: {str(e)}")
    except Exception as e:
        print(f"bartMachine package is not available: {str(e)}")
    
    # Shutdown the gateway
    print("\nShutting down gateway...")
    gateway.shutdown()

if __name__ == "__main__":
    main()
