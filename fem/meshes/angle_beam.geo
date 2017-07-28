// -*- mode: c; -*-

nelx = 200;
nely = 100;
lc = 0.01;
// Points
Point(1) = {0, 0, 0, lc};
Point(2) = {10, 4, 0, lc};
Point(3) = {10, 2, 0, lc};
Point(4) = {0, 4, 0, lc};

// Lines
Line(1) = {1, 3};
Line(2) = {3, 2};
Line(3) = {2, 4};
Line(4) = {4, 1};

// Surface 
Line Loop(5) = {3, 4, 1, 2};
Plane Surface(6) = {5};
Transfinite Line {3, 1} = nelx;//21 Using Progression 1;
Transfinite Line {4, 2} = nely;//11 Using Progression 1;
Transfinite Surface {6};
Recombine Surface {6};

// Physical groups
Physical Surface(100) = {6};
Physical Line(300) = {4}; 
Physical Line(400) = {4};  // 
Physical Line(500) = {2};  // force

