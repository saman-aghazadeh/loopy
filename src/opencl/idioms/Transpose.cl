//
// (c) July 19, 2018 Saman Biookaghazadeh @ Arizona State University
//
// Considerations: 
// This code works on a two dimensional dataset
// The host code needs to make sure that the horizontal
// and vertical size of the data is similar. That means
// get_global_size(0) and get_global_size(1) both return
// the same value.
//
//

__kernel void Transpose (__global const int* restrict A,
												 __global int* restrict B)
{

		const int gidX = get_global_id(0);
    const int gidY = get_global_id(1);
		const int lengthX = get_global_size(0);

		B[gidY*lengthX + gidX] = A[gidX*lengthX + gidY];

}