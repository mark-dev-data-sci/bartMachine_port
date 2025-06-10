"""
Check if the BartMachineWrapper class is properly loaded in the JAR file.
"""

from bartmachine.zzz import initialize_jvm, shutdown_jvm, _gateway

def main():
    # Initialize JVM
    print("Initializing JVM...")
    initialize_jvm(debug=True)

    try:
        # Get the gateway
        gateway = _gateway
        if gateway is None:
            print("Gateway is None. JVM might not be initialized properly.")
            return

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
                
                # Try to instantiate the class
                print("\nTrying to instantiate BartMachineWrapper class...")
                try:
                    wrapper_instance = wrapper_class()
                    print("Successfully instantiated BartMachineWrapper class.")
                    print(f"Instance: {wrapper_instance}")
                except Exception as e:
                    print(f"Failed to instantiate BartMachineWrapper class: {str(e)}")
            except Exception as e:
                print(f"BartMachineWrapper class is not available: {str(e)}")
        except Exception as e:
            print(f"bartMachine package is not available: {str(e)}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        # Shutdown JVM
        print("\nShutting down JVM...")
        shutdown_jvm()

if __name__ == "__main__":
    main()
