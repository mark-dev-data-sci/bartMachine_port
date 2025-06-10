#!/bin/bash

# This script compiles the BartMachineWrapper.java file and updates the JAR file

# Set the current directory to the script directory
cd "$(dirname "$0")"

# Create a temporary directory for compilation
mkdir -p temp
cd temp

# Extract the JAR file
echo "Extracting JAR file..."
jar xf ../bart_java.jar

# Copy the BartMachineWrapper.java file to the correct package directory
echo "Copying BartMachineWrapper.java..."
mkdir -p bartMachine
cp ../src/bartMachine/BartMachineWrapper.java bartMachine/

# Compile the BartMachineWrapper.java file
echo "Compiling BartMachineWrapper.java..."
javac -cp "../bart_java.jar:../commons-math-2.1.jar:../fastutil-core-8.5.8.jar:../trove-3.0.3.jar" bartMachine/BartMachineWrapper.java

# Update the JAR file
echo "Updating JAR file..."
jar uf ../bart_java.jar bartMachine/BartMachineWrapper.class

# Clean up
echo "Cleaning up..."
cd ..
rm -rf temp

echo "Done!"
