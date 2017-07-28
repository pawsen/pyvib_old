#!/bin/sh

# mkdir gmsh
# tar zxvf gmsh*.tgz -C gmsh --strip-components=1
tar zxvf gmsh*.tgz 
cd gmsh-3.0.3-Linux
sudo cp bin/gmsh /usr/bin/
sudo cp -r share/doc/gmsh /usr/share/doc/
sudo cp share/man/man1/gmsh.1 /usr/share/man/man1/
