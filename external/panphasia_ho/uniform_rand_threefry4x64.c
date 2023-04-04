#include "threefry.h"
/*
Copyright 2010-2011, D. E. Shaw Research.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of D. E. Shaw Research nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#define NUMBER_THREEFRY_ROUNDS 20

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <fftw3-mpi.h>
#include "PAN_FFTW3.h"
#include "panphasia_functions.h"

threefry4x64_ctr_t arj_threefry4x64(size_t, threefry4x64_ctr_t, threefry4x64_key_t);

threefry4x64_ctr_t inverse_arj_threefry4x64(size_t, threefry4x64_ctr_t, threefry4x64_key_t);

const threefry4x64_key_t key_constant = {{0x0, 0x0, 0x0, 0x0}}; // Set key to zero for Panphasia field

threefry4x64_key_t panphasia_key;

int panphasia_key_initialised = -1;

extern const unsigned long long int p_order;
extern const int Nbasis;

extern const int parent_info[];
extern const int new_randoms[];

extern int verbose_warnings_only;

// Global variables recording descriptor parameters //////////////

extern size_t descriptor_order;
extern size_t descriptor_base_level;
extern size_t descriptor_xorigin, descriptor_yorigin, descriptor_zorigin;
extern size_t descriptor_base_size;
extern size_t descriptor_kk_limit;
extern long long int descriptor_check_digit;
extern char descriptor_name[];
extern char full_descriptor[];

extern size_t descriptor_read_in;

// Record relative coordinates for a particular descriptor

extern size_t rel_level;
extern size_t rel_origin_x, rel_origin_y, rel_origin_z;
extern size_t rel_coord_max;
////////////////////////////////////////////////////////////////////

PAN_REAL root_cell_parent_info[300]; // Able to store up to 10th order
size_t root_cell_initialised = 0;

void threefry4x64_test_(int verbose)
{

  int pass = 0;

  /* Test three examples taken from file kat_vectors supplied with version 1.09 */

  {
    threefry4x64_ctr_t ctr = {{0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000}};
    ;
    threefry4x64_key_t key = {{0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000}};
    threefry4x64_ctr_t result = {{0x09218ebde6c85537, 0x55941f5266d86105, 0x4bd25e16282434dc, 0xee29ec846bd2e40b}};
    threefry4x64_ctr_t rand;

    rand = threefry4x64_R(NUMBER_THREEFRY_ROUNDS, ctr, key);
    if ((rand.v[0] != result.v[0]) || (rand.v[1] != result.v[1]) || (rand.v[2] != result.v[2]) || (rand.v[3] != result.v[3]))
    {
      printf("Serious error occured !!!!!!!!!!  Random generator is not working correctly \n");
      printf("Random generated: %llu %llu %llu %llu\n", rand.v[0], rand.v[1], rand.v[2], rand.v[3]);
      printf("Random expected:  %llu %llu %llu %llu\n", result.v[0], result.v[1], result.v[2], result.v[3]);
      //abort();
    }
    else
      pass++;
  }

  {
    threefry4x64_ctr_t ctr = {{0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff}};
    ;
    threefry4x64_key_t key = {{0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff}};
    ;
    threefry4x64_ctr_t result = {{0x29c24097942bba1b, 0x0371bbfb0f6f4e11, 0x3c231ffa33f83a1c, 0xcd29113fde32d168}};
    threefry4x64_ctr_t rand;

    rand = threefry4x64_R(NUMBER_THREEFRY_ROUNDS, ctr, key);

    if ((rand.v[0] != result.v[0]) || (rand.v[1] != result.v[1]) || (rand.v[2] != result.v[2]) || (rand.v[3] != result.v[3]))
    {
      printf("Serious error occured !!!!!!!!!!  Random generator is not working correctly \n");
      printf("Random generated: %llu %llu %llu %llu\n", rand.v[0], rand.v[1], rand.v[2], rand.v[3]);
      printf("Random expected:  %llu %llu %llu %llu\n", result.v[0], result.v[1], result.v[2], result.v[3]);
      //abort();
    }
    else
      pass++;
  }

  {
    threefry4x64_ctr_t ctr = {{0x243f6a8885a308d3, 0x13198a2e03707344, 0xa4093822299f31d0, 0x082efa98ec4e6c89}};
    ;
    threefry4x64_key_t key = {{0x452821e638d01377, 0xbe5466cf34e90c6c, 0xbe5466cf34e90c6c, 0xc0ac29b7c97c50dd}};
    ;
    threefry4x64_ctr_t result = {{0xa7e8fde591651bd9, 0xbaafd0c30138319b, 0x84a5c1a729e685b9, 0x901d406ccebc1ba4}};
    ;
    threefry4x64_ctr_t rand;

    rand = threefry4x64_R(NUMBER_THREEFRY_ROUNDS, ctr, key);

    if ((rand.v[0] != result.v[0]) || (rand.v[1] != result.v[1]) || (rand.v[2] != result.v[2]) || (rand.v[3] != result.v[3]))
    {
      printf("Serious error occured !!!!!!!!!!  Random generator is not working correctly \n");
      printf("Random generated: %llu %llu %llu %llu\n", rand.v[0], rand.v[1], rand.v[2], rand.v[3]);
      printf("Random expected:  %llu %llu %llu %llu\n", result.v[0], result.v[1], result.v[2], result.v[3]);
      //abort();
    }
    else
      pass++;
  }

  if ((verbose) && (pass == 3))
  {
    printf("***************************************************\n");
    printf("* Basic test of threefry4x64 generator successful *\n");
    printf("***************************************************\n");
  };

  if (NUMBER_THREEFRY_ROUNDS != 20)
  {
    for (int i = 0; i < 10; i++)
      printf("WARNING: ***************************************************\n");
    printf("WARNING: number of threefry4x64 rounds set to %d\n", NUMBER_THREEFRY_ROUNDS);
    for (int i = 0; i < 10; i++)
      printf("WARNING: ***************************************************\n");
  };
  return;
}

void set_panphasia_key_(int verbose)
{
  panphasia_key = key_constant;

  verbose = 0; //ARJ

  if (verbose)
    printf("Setting the threefry4x64 key to\n(%0llu %0llu %0llu %0llu)\n\n",
           panphasia_key.v[0], panphasia_key.v[1], panphasia_key.v[2], panphasia_key.v[3]);
  panphasia_key_initialised = 999;

  size_t level, j1, j2, j3;
  PAN_REAL unif_randoms[8 * Nbasis];
  PAN_REAL gauss_randoms[8 * Nbasis];
  PAN_REAL legendre_randoms[8 * Nbasis];
  PAN_REAL parent[Nbasis]; // Should not be used //

  // Select special pair of values to turn on function in return_uniform_pseudo_rands_threefry4x64_
  size_t seed_value = 1000000000999;
  size_t allow_non_zero_seed_saftey_catch = 1002003004005006007;

  for (int i = 0; i < Nbasis; i++)
    parent[i] = 0;

  // These parameters are ignored but need to pass error checking tests in  return_uniform... //
  level = 0;
  j1 = 0;
  j2 = 0;
  j3 = 0;

  return_uniform_pseudo_rands_threefry4x64_(level, j1, j2, j3, unif_randoms, seed_value, allow_non_zero_seed_saftey_catch);
  box_muller_(unif_randoms, gauss_randoms);
  solve_panphasia_cell_(parent, gauss_randoms, legendre_randoms, -999); //-999 no constraints

  for (int i = 0; i < Nbasis; i++)
    root_cell_parent_info[i] = legendre_randoms[i];

  root_cell_initialised = 999999;

  if (verbose_warnings_only != 1)
  {
    printf("Root cell coefficients:\n");
    for (int i = 0; i < 10; i++)
      printf("%d %15.9f\n", i, root_cell_parent_info[i]);
  };
}

void return_root_legendre_coefficients_(PAN_REAL *root)
{
  if (root_cell_initialised != 999999)
  {
    printf("Call to return_root_legendre_coefficients_ before root cell initialised\n");
    abort();
  };
  for (int i = 0; i < Nbasis; i++)
    root[i] = root_cell_parent_info[i];
}

void check_panphasia_key_(int verbose)
{

  threefry4x64_key_t panphasia_check_key;

  panphasia_check_key = key_constant;

  if (panphasia_check_key.v[0] != panphasia_key.v[0] || panphasia_check_key.v[1] != panphasia_key.v[1] || panphasia_check_key.v[2] != panphasia_key.v[2] || panphasia_check_key.v[2] != panphasia_key.v[2])
  {
    printf("A serious error has happened - the threefry4x64 key has become corrupted!\n");
    printf("Should be:  (%0llu %0llu %0llu %0llu)\n", panphasia_check_key.v[0],
           panphasia_check_key.v[1], panphasia_check_key.v[2], panphasia_check_key.v[3]);

    printf("But now is: (%0llu %0llu %0llu %0llu)\n", panphasia_key.v[0],
           panphasia_key.v[1], panphasia_key.v[2], panphasia_key.v[3]);
    printf("The fact that it has changed suggests the key has been overwritten in memory.\n");
    abort();
  };

  if (verbose)
    printf("The key has been checked and has not been corrupted\n");
}

//=============================================================================================
void return_uniform_pseudo_rands_threefry4x64_(size_t l, size_t j1, size_t j2, size_t j3,
                                               PAN_REAL *panphasia_randoms, size_t seed_value,
                                               size_t allow_non_zero_seed_saftey_catch)
{
  int i, j;
  size_t j0;

  // threefry4x64_ctr_t  ctr_base;
  threefry4x64_ctr_t ctr;
  threefry4x64_ctr_t rand;

  // int ncalls;
  int ncount;

  unsigned int out_int[8];

  PAN_REAL unif_real[8 * Nbasis]; //__assume_aligned(&unif_real, 64);

  /* These specific choices of g_scale, g_shift give RMS very close to 1  */
  double g_scale = 1.0 / 4294967296.0; //
  double g_shift = 0.5;                // To avoid the value zero for uniform random numbers

  // BEGIN ERROR CHECKING //

  if (panphasia_key_initialised != 999)
  {
    printf("Panphasia threefry4x64 key not initialised ..."); // Not safe to proceed ...
    abort();
  };

  if ((l < 0) || (l > 63))
  {
    printf("Level %lu is out of range (0-63)!\n", l); // Not part of Panphasia
    abort();
  };

  if ((j1 >> l != 0) || (j2 >> l != 0) || (j3 >> l != 0))
  { // Cell outside of Panphasia
    printf("Level %lu: Cell coordinate out of range (%lu,%lu,%lu)\n", l, j1, j2, j3);
    abort();
  };

  // Only allow a non-zero value for the seed if the safety catch has a specific value

  if (allow_non_zero_seed_saftey_catch != 1002003004005006007)
  {
    seed_value = 0;
  };

  size_t root_cell_calculation = 0;
  //=============================================================================
  // Exception - for computing the parent properties of the root cell only
  //=============================================================================
  if ((allow_non_zero_seed_saftey_catch == 1002003004005006007) && (seed_value == 1000000000999))
  {
    l = 0;
    j0 = (p_order << 60);
    j1 = 2;
    j2 = 2;
    j3 = 2;
    if (p_order > 8)
    {
      printf("Multipole order too high\n");
      abort();
    };
    seed_value = 0;
    root_cell_calculation = 1; //Signal root cell properties are being calculated
  };
  //===================================================

  if (seed_value >> 32 != 0)
  {
    printf("Seed value %lu, outside range   0 <= seed <2^32 \n", seed_value);
    abort();
  };

  // END ERROR CHECKING //

  int nloop = Nbasis; // Generate eight uniform randoms per call of Threefry4x64 //

  size_t k0, k1, k2, k3;

  j0 = (p_order << 60) + ((l << 56) >> 4) + ((seed_value << 32) >> 12);

  k0 = j0;
  k1 = j1;
  k2 = j2;
  k3 = j3;

  if ((root_cell_calculation) && (verbose_warnings_only != 1))
  {
    printf("============================================================================================\n");
    printf("Computing root cell properties\n");
    printf("p_order, l, seed_value: (j0,j1,j2,j3),%llx %lx %lx (%lx,%lx,%lx,%lx)\n",
           p_order, l, seed_value, j0, j1, j2, j3);
    printf("Encoded root cell values:(k0,k1,k2,k3):\n        (%lx,%lx,%lx,%lx)\n", k0, k1, k2, k3);
    printf("============================================================================================\n");
  };

  ctr.v[0] = k0;
  ctr.v[1] = k1;
  ctr.v[2] = k2;
  ctr.v[3] = k3;

  ncount = 0;

  for (i = 0; i < nloop; ++i)
  {

    rand = threefry4x64_R(NUMBER_THREEFRY_ROUNDS, ctr, panphasia_key);

    ctr.v[0] += 1;

    out_int[0] = rand.v[0] >> 32;
    out_int[1] = (rand.v[0] << 32) >> 32;

    out_int[2] = rand.v[1] >> 32;
    out_int[3] = (rand.v[1] << 32) >> 32;

    out_int[4] = rand.v[2] >> 32;
    out_int[5] = (rand.v[2] << 32) >> 32;

    out_int[6] = rand.v[3] >> 32;
    out_int[7] = (rand.v[3] << 32) >> 32;

    for (j = 0; j < 8; ++j)
      unif_real[ncount++] = (((double)out_int[j] + g_shift) * g_scale);
  };

  for (i = 0; i < 8 * Nbasis; i++)
    panphasia_randoms[i] = unif_real[i];

  // Exceptional branch with the aim ultimately of filling the Gaussian tail.
  // Executed rarely so does not need to be particularly efficient.
  // For this reason it include an error check. Can the value
  // that triggered this loop be reproduced? If it cannot, the code aborts.

  size_t branch_value = 4096;
  PAN_REAL branching_ratio = (((double)branch_value) * g_scale);

  //PAN_REAL branching_ratio = -0.3;

  for (size_t i = 0; i < 8 * Nbasis; i += 2)
    if (panphasia_randoms[i] < branching_ratio)
    {

      PAN_REAL new_value;
      PAN_REAL replacement_value = branching_ratio;

      for (size_t loop = 0; loop < 2; loop++)
      {                                            //Loop==0, check can reproduce result //Loop=1 new random
                                                   //Loop==1, make sure randoms cannot be duplicated.
        size_t iind = i / 8 + loop * (Nbasis + i); // Skip beyond sequence previously used.
        size_t jind = i % 8;

        j0 = (p_order << 60) + ((l << 56) >> 4) + ((seed_value << 32) >> 12);
        //code_cell(j0,j1,j2,j3,&k0,&k1,&k2,&k3);

        k0 = j0;
        k1 = j1;
        k2 = j2;
        k3 = j3;

        ctr.v[0] = k0 + iind;
        ctr.v[1] = k1;
        ctr.v[2] = k2;
        ctr.v[3] = k3;

        //          ctr.v[0] = k0+iind*increment;
        // ctr.v[1] = k1+iind*increment;
        //ctr.v[2] = k2+iind*increment;
        //ctr.v[3] = k3+iind*increment;

        rand = threefry4x64_R(NUMBER_THREEFRY_ROUNDS, ctr, panphasia_key);

        out_int[0] = rand.v[0] >> 32;
        out_int[1] = (rand.v[0] << 32) >> 32;

        out_int[2] = rand.v[1] >> 32;
        out_int[3] = (rand.v[1] << 32) >> 32;

        out_int[4] = rand.v[2] >> 32;
        out_int[5] = (rand.v[2] << 32) >> 32;

        out_int[6] = rand.v[3] >> 32;
        out_int[7] = (rand.v[3] << 32) >> 32;

        new_value = (((double)out_int[jind] + g_shift) * g_scale);

        if (loop == 0)
        {
          if (new_value != panphasia_randoms[i])
          {
            printf("Failure to reproduce the initial random that triggered this branch - a serious error!\n");
            abort();
          }
        }
        else
        {

          if (new_value >= branching_ratio)
          {

            replacement_value *= new_value;
          }
          else
          {

            size_t counter = 0;

            while ((new_value < branching_ratio) && (counter < 7))
            {
              replacement_value *= branching_ratio;
              counter++;
              // jind = (++jind) % 8;
              jind = (jind+1)%8;
              new_value = (((double)out_int[jind] + g_shift) * g_scale);
            };
            replacement_value *= new_value;
            //if (new_value<branching_ratio) {

            //  replacement_value=0.5; // Not in the tail!!!

            //  printf("Warning - not enough branches! Cell: %llx %llu %llu %llu i=%llu \n",j0,j1,j2,j3,i);
            //      printf("Value of panphasia_randoms[i] %e\n",panphasia_randoms[i]);
            //};
          };
        };

      }; // End loop

      panphasia_randoms[i] = replacement_value;
    };

  return;
}

//===========================================================//

#include <string.h>

long long int compute_check_digit_()
{

  char str[200];
  long long int check_digit;

  threefry4x64_ctr_t ctr, rand;
  threefry4x64_key_t key;

  if (descriptor_read_in == 0)
  {
    printf("No descriptor has been set\n");
    abort();
  };

  sprintf(str, "%lu%lu%lu%lu%lu%lu%lu%s", descriptor_order, descriptor_base_level,
          descriptor_xorigin, descriptor_yorigin, descriptor_zorigin,
          descriptor_base_size, descriptor_kk_limit, descriptor_name);

  key = key_constant;

  ctr.v[0] = 0;
  ctr.v[1] = 0;
  ctr.v[2] = 0;
  ctr.v[3] = 0;

  for (int i = 0; i < strlen(str); i++)
  {
    ctr.v[0] = ctr.v[0] + (int)str[i];
    rand = threefry4x64(ctr, key);
    ctr = rand;
  };

  check_digit = (ctr.v[0] >> 32);

  return (check_digit);
}

//////////////////////////////////////////////////////////////////////////////
// Construct pairs of overlapping random descriptors for testing
// moments of a cell and its child eight child cells are
// essentially identical.
//////////////////////////////////////////////////////////////////////////////

void test_propogation_of_moments_(int iterations)
{

  const int level_max = 62;

  threefry4x64_ctr_t ctr, rand;
  threefry4x64_key_t key;

  key = key_constant;

  ctr.v[0] = 0;
  ctr.v[1] = 0;
  ctr.v[2] = 0;
  ctr.v[3] = 0;

  int levplus = 1;
  if (iterations == 0)
  {
    iterations = 1;
    levplus = 5;
  };

  for (int it = 0; it < iterations; it++)
  {

    for (int lev = 0; lev < 63; lev += levplus)
    {

      size_t level_desc1 = lev;

      rand = threefry4x64(ctr, key);
      ctr = rand;

      size_t side_length1 = ((size_t)1 << level_desc1);

      size_t level_cell;
      size_t level_desc2;

      if (level_max - level_desc1 > 0)
      {
        level_cell = level_desc1 + ctr.v[0] % (level_max - level_desc1);
      }
      else
      {
        level_cell = level_desc1;
      };

      if (level_cell - level_desc1 > 0)
      {
        level_desc2 = level_desc1 + ctr.v[1] % (level_cell - level_desc1);
      }
      else
      {
        level_desc2 = level_desc1;
      };

      size_t side_length2 = (size_t)1 << level_desc2;
      size_t side_length3 = (size_t)1 << level_cell;

      rand = threefry4x64(ctr, key);
      ctr = rand;

      size_t xcell = ctr.v[0] >> (64 - level_cell);
      size_t ycell = ctr.v[1] >> (64 - level_cell);
      size_t zcell = ctr.v[2] >> (64 - level_cell);

      //size_t cell_level_size = (size_t)1<<level_cell;

      rand = threefry4x64(ctr, key);
      ctr = rand;

      size_t desc1_s = ctr.v[0] % side_length1;
      size_t desc2_s = ctr.v[1] % side_length2;

      if (desc1_s == 0)
        desc1_s = 1; // Enforce minimum descriptor size of 1
      if (desc2_s == 0)
        desc2_s = 1; // Enforce minimum descriptor size of 1

      rand = threefry4x64(ctr, key);
      ctr = rand;

      size_t dx1 = ctr.v[0] % desc1_s;
      size_t dy1 = ctr.v[1] % desc1_s;
      size_t dz1 = ctr.v[2] % desc1_s;

      rand = threefry4x64(ctr, key);
      ctr = rand;

      size_t dx2 = ctr.v[0] % desc2_s;
      size_t dy2 = ctr.v[1] % desc2_s;
      size_t dz2 = ctr.v[2] % desc2_s;

      size_t desc1_x = ((xcell >> (level_cell - level_desc1)) - dx1 + side_length1) % side_length1;
      size_t desc1_y = ((ycell >> (level_cell - level_desc1)) - dy1 + side_length1) % side_length1;
      size_t desc1_z = ((zcell >> (level_cell - level_desc1)) - dz1 + side_length1) % side_length1;

      size_t desc2_x = ((xcell >> (level_cell - level_desc2)) - dx2 + side_length2) % side_length2;
      size_t desc2_y = ((ycell >> (level_cell - level_desc2)) - dy2 + side_length2) % side_length2;
      size_t desc2_z = ((zcell >> (level_cell - level_desc2)) - dz2 + side_length2) % side_length2;

      char descriptor1[300];
      char descriptor2[300];

      sprintf(descriptor1, "[Panph%lld,L%ld,(%lu,%lu,%lu),S%lu,CH-999,test]",
              p_order, level_desc1, desc1_x, desc1_y, desc1_z, desc1_s);

      sprintf(descriptor2, "[Panph%lld,L%ld,(%lu,%lu,%lu),S%lu,CH-999,test]",
              p_order, level_desc2, desc2_x, desc2_y, desc2_z, desc2_s);

      //printf("Descriptor 1: %s\nDescriptor 2: %s\n",descriptor1,descriptor2);

      rand = threefry4x64(ctr, key);
      ctr = rand;

      size_t rel_level1 = level_cell - level_desc1;
      size_t rel_level2 = level_cell - level_desc2;

      size_t xstart1 = (xcell - (desc1_x << rel_level1) + side_length3) % side_length3;
      size_t ystart1 = (ycell - (desc1_y << rel_level1) + side_length3) % side_length3;
      size_t zstart1 = (zcell - (desc1_z << rel_level1) + side_length3) % side_length3;

      size_t xstart2 = (xcell - (desc2_x << rel_level2) + side_length3) % side_length3;
      size_t ystart2 = (ycell - (desc2_y << rel_level2) + side_length3) % side_length3;
      size_t zstart2 = (zcell - (desc2_z << rel_level2) + side_length3) % side_length3;

      size_t extent1 = 1;
      size_t extent2 = 1;

      if (ctr.v[0] % 2 == 0)
      {
        rel_level1++;
        xstart1 *= 2;
        ystart1 *= 2;
        zstart1 *= 2;
        extent1 *= 2;
      }
      else
      {
        rel_level2++;
        xstart2 *= 2;
        ystart2 *= 2;
        zstart2 *= 2;
        extent2 *= 2;
      };

      //printf("1: relative level %llu  Rel offset: (%llu %llu %llu) Extent %llu\n",rel_level1,xstart1,
      //	ystart1,zstart1,extent1);

      //printf("2: relative level %llu  Rel offset: (%llu %llu %llu) Extent %llu\n",rel_level2,xstart2,
      //	ystart2,zstart2,extent2);

      verbose_warnings_only = 1; // Minimize output to screen.

      double max_diff2 = 0.0;
      double rms_diff2 = 0.0;

      double coefficients1[Nbasis];
      double coefficients2[Nbasis];

      test_cell_moments(descriptor1, rel_level1, xstart1, ystart1, zstart1, extent1, coefficients1);
      test_cell_moments(descriptor2, rel_level2, xstart2, ystart2, zstart2, extent2, coefficients2);

      for (int i = 0; i < Nbasis; i++)
      {
        double diff2 = pow(coefficients2[i] - coefficients1[i], 2);
        if (diff2 > max_diff2)
          max_diff2 = diff2;
        rms_diff2 += diff2;
      };

      // printf("%s\n%s\n",descriptor1,descriptor2);

      //printf("Example coeff %18.12lf %18.12lf \n",coefficients1[0],coefficients2[0]);

      rms_diff2 /= (double)Nbasis;

      if ((sizeof(PAN_REAL) == 4) || (sizeof(FFTW_REAL) == 4))
      {

        if ((max_diff2 > 1.e-12) || (rms_diff2 > 1.e-12))
        {
          printf("Moments not accurately recovered at single precision\n");
          abort();
        };
      }
      else
      {

        if ((max_diff2 > 1.e-24) || (rms_diff2 > 1.e-24))
        {
          printf("Moments not accurately recovered at double precision\n");
          abort();
        };
      };

      // printf("lev %d Acceptable differences:  %e   RMS difference %e\n",lev,sqrt(max_diff2),sqrt(rms_diff2));
    };

    //  printf("Test of descriptors/relative coordinates and moments PASSED.\n");
  };
}

/////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
//  Alternative Threefry4x64 generator and inverse function - for testing
//  purposes only
/////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

void inverse_threefry4x64_test_(int verbose)
{

  threefry4x64_ctr_t ctr = {{0x243f6a8885a308d3, 0x13198a2e03707344, 0xa4093822299f31d0, 0x082efa98ec4e6c89}};
  ;
  threefry4x64_key_t key = {{0x452821e638d01377, 0xbe5466cf34e90c6c, 0xbe5466cf34e90c6c, 0xc0ac29b7c97c50dd}};
  ;
  threefry4x64_ctr_t rand1, rand2;

  for (size_t ROUNDS = 0; ROUNDS < 21; ROUNDS++)
  {

    rand1 = threefry4x64_R(ROUNDS, ctr, key);

    rand2 = arj_threefry4x64(ROUNDS, ctr, key);

    if ((rand1.v[0] != rand2.v[0]) || (rand1.v[1] != rand2.v[1]) || (rand1.v[2] != rand2.v[2]) || (rand1.v[3] != rand2.v[3]))
    {
      printf("Error in arj_threefry4x64 - failing to reproduce Threefry4x64 generator!!!\n");
      abort();
    };

    rand2 = inverse_arj_threefry4x64(ROUNDS, rand1, key);

    if ((ctr.v[0] != rand2.v[0]) || (ctr.v[1] != rand2.v[1]) || (ctr.v[2] != rand2.v[2]) || (ctr.v[3] != rand2.v[3]))
    {
      printf("Error in arj_threefry4x64 - failing to reproduce INVERSE Threefry4x64 generator!!!\n");
      abort();
    };
  };

  return;
}

threefry4x64_ctr_t arj_threefry4x64(size_t R, threefry4x64_ctr_t ctr,
                                    threefry4x64_key_t key)
{
  size_t x0 = ctr.v[0];
  size_t x1 = ctr.v[1];
  size_t x2 = ctr.v[2];
  size_t x3 = ctr.v[3];
  size_t k0 = key.v[0];
  size_t k1 = key.v[1];
  size_t k2 = key.v[2];
  size_t k3 = key.v[3];
  size_t k4 = 0x1bd11bdaa9fc1a22;
  //---------------------------------------

  if (R > 20)
    abort();

  k4 ^= k0;
  k4 ^= k1;
  k4 ^= k2;
  k4 ^= k3;
  x0 += k0;
  x1 += k1;
  x2 += k2;
  x3 += k3;

  if (R > 0)
  {
    x0 += x1;
    x1 = (x1 << 14) | (x1 >> 50);
    x1 ^= x0;
    x2 += x3;
    x3 = (x3 << 16) | (x3 >> 48);
    x3 ^= x2;
  };
  if (R > 1)
  {
    x0 += x3;
    x3 = (x3 << 52) | (x3 >> 12);
    x3 ^= x0;
    x2 += x1;
    x1 = (x1 << 57) | (x1 >> 7);
    x1 ^= x2;
  };

  if (R > 2)
  {
    x0 += x1;
    x1 = (x1 << 23) | (x1 >> 41);
    x1 ^= x0;
    x2 += x3;
    x3 = (x3 << 40) | (x3 >> 24);
    x3 ^= x2;
  };

  if (R > 3)
  {
    x0 += x3;
    x3 = (x3 << 5) | (x3 >> 59);
    x3 ^= x0;
    x2 += x1;
    x1 = (x1 << 37) | (x1 >> 27);
    x1 ^= x2;
    //Inject key 1
    x0 += k1;
    x1 += k2;
    x2 += k3;
    x3 += k4;
    x3 += 1;
  };

  if (R > 4)
  {
    x0 += x1;
    x1 = (x1 << 25) | (x1 >> 39);
    x1 ^= x0;
    x2 += x3;
    x3 = (x3 << 33) | (x3 >> 31);
    x3 ^= x2;
  };

  if (R > 5)
  {
    x0 += x3;
    x3 = (x3 << 46) | (x3 >> 18);
    x3 ^= x0;
    x2 += x1;
    x1 = (x1 << 12) | (x1 >> 52);
    x1 ^= x2;
  };

  if (R > 6)
  {
    x0 += x1;
    x1 = (x1 << 58) | (x1 >> 6);
    x1 ^= x0;
    x2 += x3;
    x3 = (x3 << 22) | (x3 >> 42);
    x3 ^= x2;
  };

  if (R > 7)
  {
    x0 += x3;
    x3 = (x3 << 32) | (x3 >> 32);
    x3 ^= x0;
    x2 += x1;
    x1 = (x1 << 32) | (x1 >> 32);
    x1 ^= x2;
    //Inject key 2
    x0 += k2;
    x1 += k3;
    x2 += k4;
    x3 += k0;
    x3 += 2;
  };

  if (R > 8)
  {
    x0 += x1;
    x1 = (x1 << 14) | (x1 >> 50);
    x1 ^= x0;
    x2 += x3;
    x3 = (x3 << 16) | (x3 >> 48);
    x3 ^= x2;
  };

  if (R > 9)
  {
    x0 += x3;
    x3 = (x3 << 52) | (x3 >> 12);
    x3 ^= x0;
    x2 += x1;
    x1 = (x1 << 57) | (x1 >> 7);
    x1 ^= x2;
  };

  if (R > 10)
  {
    x0 += x1;
    x1 = (x1 << 23) | (x1 >> 41);
    x1 ^= x0;
    x2 += x3;
    x3 = (x3 << 40) | (x3 >> 24);
    x3 ^= x2;
  };

  if (R > 11)
  {
    x0 += x3;
    x3 = (x3 << 5) | (x3 >> 59);
    x3 ^= x0;
    x2 += x1;
    x1 = (x1 << 37) | (x1 >> 27);
    x1 ^= x2;
    //Inject key 3
    x0 += k3;
    x1 += k4;
    x2 += k0;
    x3 += k1;
    x3 += 3;
  };

  if (R > 12)
  {
    x0 += x1;
    x1 = (x1 << 25) | (x1 >> 39);
    x1 ^= x0;
    x2 += x3;
    x3 = (x3 << 33) | (x3 >> 31);
    x3 ^= x2;
  };

  if (R > 13)
  {
    x0 += x3;
    x3 = (x3 << 46) | (x3 >> 18);
    x3 ^= x0;
    x2 += x1;
    x1 = (x1 << 12) | (x1 >> 52);
    x1 ^= x2;
  };

  if (R > 14)
  {
    x0 += x1;
    x1 = (x1 << 58) | (x1 >> 6);
    x1 ^= x0;
    x2 += x3;
    x3 = (x3 << 22) | (x3 >> 42);
    x3 ^= x2;
  };

  if (R > 15)
  {
    x0 += x3;
    x3 = (x3 << 32) | (x3 >> 32);
    x3 ^= x0;
    x2 += x1;
    x1 = (x1 << 32) | (x1 >> 32);
    x1 ^= x2;
    //Inject key 4
    x0 += k4;
    x1 += k0;
    x2 += k1;
    x3 += k2;
    x3 += 4;
  };

  if (R > 16)
  {
    x0 += x1;
    x1 = (x1 << 14) | (x1 >> 50);
    x1 ^= x0;
    x2 += x3;
    x3 = (x3 << 16) | (x3 >> 48);
    x3 ^= x2;
  };

  if (R > 17)
  {
    x0 += x3;
    x3 = (x3 << 52) | (x3 >> 12);
    x3 ^= x0;
    x2 += x1;
    x1 = (x1 << 57) | (x1 >> 7);
    x1 ^= x2;
  };
  if (R > 18)
  {
    x0 += x1;
    x1 = (x1 << 23) | (x1 >> 41);
    x1 ^= x0;
    x2 += x3;
    x3 = (x3 << 40) | (x3 >> 24);
    x3 ^= x2;
  };
  if (R > 19)
  {
    x0 += x3;
    x3 = (x3 << 5) | (x3 >> 59);
    x3 ^= x0;
    x2 += x1;
    x1 = (x1 << 37) | (x1 >> 27);
    x1 ^= x2;
    //Inject key 5
    x0 += k0;
    x1 += k1;
    x2 += k2;
    x3 += k3;
    x3 += 5;
  };
  //---------------------------------------
  threefry4x64_ctr_t result = {{x0, x1, x2, x3}};
  return (result);
}

threefry4x64_ctr_t inverse_arj_threefry4x64(size_t R, threefry4x64_ctr_t ctr,
                                            threefry4x64_key_t key)
{

  size_t x0 = ctr.v[0];
  size_t x1 = ctr.v[1];
  size_t x2 = ctr.v[2];
  size_t x3 = ctr.v[3];
  size_t k0 = key.v[0];
  size_t k1 = key.v[1];
  size_t k2 = key.v[2];
  size_t k3 = key.v[3];
  size_t k4 = 0x1bd11bdaa9fc1a22;
  //---------------------------------------

  if (R > 20)
    abort();

  k4 ^= k0;
  k4 ^= k1;
  k4 ^= k2;
  k4 ^= k3;

  if (R > 19)
  {
    //Anti-inject key 5
    x0 -= k0;
    x1 -= k1;
    x2 -= k2;
    x3 -= k3;
    x3 -= 5;
    x3 ^= x0;
    x3 = (x3 << 59) | (x3 >> 5);
    x0 -= x3;
    x1 ^= x2;
    x1 = (x1 << 27) | (x1 >> 37);
    x2 -= x1;
  };

  if (R > 18)
  {
    x3 ^= x2;
    x3 = (x3 << 24) | (x3 >> 40);
    x2 -= x3;
    x1 ^= x0;
    x1 = (x1 << 41) | (x1 >> 23);
    x0 -= x1;
  };

  if (R > 17)
  {
    x3 ^= x0;
    x3 = (x3 << 12) | (x3 >> 52);
    x0 -= x3;
    x1 ^= x2;
    x1 = (x1 << 7) | (x1 >> 57);
    x2 -= x1;
  };

  if (R > 16)
  {
    x3 ^= x2;
    x3 = (x3 << 48) | (x3 >> 16);
    x2 -= x3;
    x1 ^= x0;
    x1 = (x1 << 50) | (x1 >> 14);
    x0 -= x1;
  };

  if (R > 15)
  {
    //Anti-inject key 4
    x0 -= k4;
    x1 -= k0;
    x2 -= k1;
    x3 -= k2;
    x3 -= 4;
    x3 ^= x0;
    x3 = (x3 << 32) | (x3 >> 32);
    x0 -= x3;
    x1 ^= x2;
    x1 = (x1 << 32) | (x1 >> 32);
    x2 -= x1;
  };

  if (R > 14)
  {
    x3 ^= x2;
    x3 = (x3 << 42) | (x3 >> 22);
    x2 -= x3;
    x1 ^= x0;
    x1 = (x1 << 6) | (x1 >> 58);
    x0 -= x1;
  };

  if (R > 13)
  {
    x3 ^= x0;
    x3 = (x3 << 18) | (x3 >> 46);
    x0 -= x3;
    x1 ^= x2;
    x1 = (x1 << 52) | (x1 >> 12);
    x2 -= x1;
  };

  if (R > 12)
  {
    x3 ^= x2;
    x3 = (x3 << 31) | (x3 >> 33);
    x2 -= x3;
    x1 ^= x0;
    x1 = (x1 << 39) | (x1 >> 25);
    x0 -= x1;
  };

  if (R > 11)
  {
    //Anti-inject key 3
    x0 -= k3;
    x1 -= k4;
    x2 -= k0;
    x3 -= k1;
    x3 -= 3;
    x3 ^= x0;
    x3 = (x3 << 59) | (x3 >> 5);
    x0 -= x3;
    x1 ^= x2;
    x1 = (x1 << 27) | (x1 >> 37);
    x2 -= x1;
  };

  if (R > 10)
  {
    x3 ^= x2;
    x3 = (x3 << 24) | (x3 >> 40);
    x2 -= x3;
    x1 ^= x0;
    x1 = (x1 << 41) | (x1 >> 23);
    x0 -= x1;
  };

  if (R > 9)
  {
    x3 ^= x0;
    x3 = (x3 << 12) | (x3 >> 52);
    x0 -= x3;
    x1 ^= x2;
    x1 = (x1 << 7) | (x1 >> 57);
    x2 -= x1;
  };

  if (R > 8)
  {
    x3 ^= x2;
    x3 = (x3 << 48) | (x3 >> 16);
    x2 -= x3;
    x1 ^= x0;
    x1 = (x1 << 50) | (x1 >> 14);
    x0 -= x1;
  };

  if (R > 7)
  {
    //Anti-inject key 2
    x0 -= k2;
    x1 -= k3;
    x2 -= k4;
    x3 -= k0;
    x3 -= 2;
    x3 ^= x0;
    x3 = (x3 << 32) | (x3 >> 32);
    x0 -= x3;
    x1 ^= x2;
    x1 = (x1 << 32) | (x1 >> 32);
    x2 -= x1;
  };

  if (R > 6)
  {
    x3 ^= x2;
    x3 = (x3 << 42) | (x3 >> 22);
    x2 -= x3;
    x1 ^= x0;
    x1 = (x1 << 6) | (x1 >> 58);
    x0 -= x1;
  };

  if (R > 5)
  {
    x3 ^= x0;
    x3 = (x3 << 18) | (x3 >> 46);
    x0 -= x3;
    x1 ^= x2;
    x1 = (x1 << 52) | (x1 >> 12);
    x2 -= x1;
  };

  if (R > 4)
  {
    x3 ^= x2;
    x3 = (x3 << 31) | (x3 >> 33);
    x2 -= x3;
    x1 ^= x0;
    x1 = (x1 << 39) | (x1 >> 25);
    x0 -= x1;
  };

  if (R > 3)
  {
    //Anti-inject key 1
    x0 -= k1;
    x1 -= k2;
    x2 -= k3;
    x3 -= k4;
    x3 -= 1;
    x3 ^= x0;
    x3 = (x3 << 59) | (x3 >> 5);
    x0 -= x3;
    x1 ^= x2;
    x1 = (x1 << 27) | (x1 >> 37);
    x2 -= x1;
  };

  if (R > 2)
  {
    x3 ^= x2;
    x3 = (x3 << 24) | (x3 >> 40);
    x2 -= x3;
    x1 ^= x0;
    x1 = (x1 << 41) | (x1 >> 23);
    x0 -= x1;
  };

  if (R > 1)
  {
    x3 ^= x0;
    x3 = (x3 << 12) | (x3 >> 52);
    x0 -= x3;
    x1 ^= x2;
    x1 = (x1 << 7) | (x1 >> 57);
    x2 -= x1;
  };

  if (R > 0)
  {
    x3 ^= x2;
    x3 = (x3 << 48) | (x3 >> 16);
    x2 -= x3;
    x1 ^= x0;
    x1 = (x1 << 50) | (x1 >> 14);
    x0 -= x1;
  };

  // Anti-start
  x0 -= k0;
  x1 -= k1;
  x2 -= k2;
  x3 -= k3;

  //---------------------------------------
  threefry4x64_ctr_t result = {{x0, x1, x2, x3}};
  return (result);
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
