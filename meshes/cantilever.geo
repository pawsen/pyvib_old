// Cantilever example

// Mesh Size
lcd     = 0.5; // target element size
nnx    = 15;
nny    = 9; 
// quad or triangle elements
quads = 1;

// Sidelengths
rx   = 10.0;
ry   = 1.0;

// center point !
Point(1) = {0.0,0.0,0.0,lcd};
Point(2) = {rx,0.0,0.0,lcd};
Point(3) = {rx,ry,0.0,lcd};
Point(4) = {0.0,ry,0.0,lcd};

// LInes
Line(1) = {1,2};
Line(2) = {2,3}; //load (if distributed)
Line(3) = {3,4};
Line(4) = {4,1}; //support

// Line loop
Line Loop(1) = {1,2,3,4}; // Design domain


// Create surfaces
Plane Surface(1) = {1}; // Design domain


If (quads == 1)
// RHS gives the number of nodes that will be created on the line
// (this overrides any other mesh element size prescription
Transfinite Line{1} = nnx;
Transfinite Line{2} = nny;
Transfinite Line{3} = nnx;
Transfinite Line{4} = nny;


// Create the mapped surface:
// The expression-list on the right-hand-side should contain the 
// identification numbers of three or four points on the boundary of 
// the surface that define the corners of the transfinite interpolation.
Transfinite Surface{1} = {1,2,3,4};

// Recombines the triangular meshes of the surfaces into mixed triangular/quadrangular meshes.
// Optional RHS: specifies the maximum difference (in degrees) allowed between the largest angle of a quadrangle and a right angle
// a value of 0 would only accept quadrangles with right angles; 
Recombine Surface{1} = 0;
EndIf

// Label surfaces for program
Physical Surface("Design domain",1001) = {1}; // Design Domain
Physical Line("Support x-dir",1003) = {4}; // Support
Physical Line("Support y-dir",1004) = {4}; // Support
Physical Point("load",1006) = {2}; // load 


Coherence;
