/* This program is public domain */

/**
 * Adaptive sampling for reflectivity calculations.
 * 
 * This file implements adaptive Q sampling to ensure reflectivity curves are
 * adequately sampled for accurate interpolation. The algorithm works by:
 *
 * 1. Starting with an initial set of Q points
 * 2. Calculating reflectivity at these points  
 * 3. Using cubic interpolation to estimate errors between points
 * 4. Adding new points where the interpolation error exceeds tolerance
 * 5. Repeating until convergence or maximum iterations reached
 *
 * The main functions are:
 *
 * oversample_inplace() - Core adaptive sampling algorithm
 *   Performs the iterative oversampling with cubic interpolation error estimation
 *
 * autosampled_reflectivity_amplitude() - High-level interface
 *   Wrapper function compatible with the Python API that calls oversample_inplace
 *
 * The algorithm uses cubic spline interpolation between triplets of points to 
 * estimate the interpolation error. Points are added at the midpoint of intervals
 * where this error exceeds the specified tolerance.
 */

#include <iostream>
#include <complex>
#include <vector>
#include <algorithm>
#include <cmath>
#include "reflcalc.h"

typedef std::complex<double> Cplx;

// Helper function to insert values into an array at specified positions
template<typename T>
std::vector<T> insert_values(const std::vector<T>& arr, 
                             const std::vector<int>& positions,
                             const std::vector<T>& values) {
    int ninput = arr.size();
    int nvals = values.size();
    int noutput = ninput + nvals;
    
    std::vector<T> output(noutput);
    std::vector<bool> offsets(noutput, false);
    
    // Mark insertion positions (adjusted for cumulative insertions)
    for (int i = 0; i < positions.size(); i++) {
        int pos = positions[i] + i + 1; // +1 because we insert after the position
        if (pos < noutput) {
            offsets[pos] = true;
        }
    }
    
    int input_offset = 0;
    int val_index = 0;
    
    for (int i = 0; i < noutput; i++) {
        if (offsets[i]) {
            output[i] = values[val_index];
            val_index++;
        } else {
            output[i] = arr[input_offset];
            input_offset++;
        }
    }
    
    return output;
}

// Main oversampling function that returns vectors
std::pair<std::vector<double>, std::vector<Cplx>>
oversample_inplace(const int initial_points,
                   const double initial_kz[],
                   const double initial_dR[],
                   const double tolerance,
                   const int layers,
                   const double depth[],
                   const double sigma[],
                   const double rho[],
                   const double irho[],
                   const int max_sampling_iterations)
{
    // Convert input arrays to vectors for easier manipulation
    std::vector<double> new_kz(initial_kz, initial_kz + initial_points);
    std::vector<double> mapped_dR(initial_dR, initial_dR + initial_points);
    std::vector<int> rho_index(initial_points, 0); // All zeros for single dataset
    std::vector<Cplx> r(initial_points);
    
    // Calculate initial reflectivity amplitudes
    reflectivity_amplitude(layers, depth, sigma, rho, irho, 
                          initial_points, &new_kz[0], &rho_index[0], &r[0]);
    
    // Calculate initial reflectivity values |r|²
    std::vector<double> R(initial_points);
    for (int i = 0; i < initial_points; i++) {
        R[i] = std::norm(r[i]); // |r|² = r * conj(r)
    }
    
    int iterations = 0;
    int out_of_tolerance = 1;
    
    while (out_of_tolerance > 0 && iterations < max_sampling_iterations) {
        int n = new_kz.size();
        if (n < 3) break; // Need at least 3 points for cubic interpolation
        
        // Calculate differences between adjacent points
        std::vector<double> d12(n-2), d23(n-2), d13(n-2);
        std::vector<double> y1(n-2), y2(n-2), y3(n-2);
        std::vector<double> dy2(n-2);
        
        for (int i = 0; i < n-2; i++) {
            d12[i] = new_kz[i] - new_kz[i+1];     // delta between first and second points
            d23[i] = new_kz[i+1] - new_kz[i+2];  // delta between second and third points  
            d13[i] = new_kz[i] - new_kz[i+2];    // delta between first and third points
            
            y1[i] = R[i];     // first point reflectivity
            y2[i] = R[i+1];   // second point reflectivity
            y3[i] = R[i+2];   // third point reflectivity
            
            dy2[i] = mapped_dR[i+1]; // second point mapped dR
        }
        
        // Calculate cubic interpolation coefficients
        std::vector<double> common(n-2), a1(n-2), a2(n-2);
        for (int i = 0; i < n-2; i++) {
            common[i] = (y1[i] * d23[i] - y2[i] * d13[i] + y3[i] * d12[i]) / (6.0 * d13[i]);
            a1[i] = d12[i] / d23[i] * common[i];
            a2[i] = d23[i] / d12[i] * common[i];
        }
        
        // Calculate intervals between adjacent kz points
        std::vector<double> dx(n-1);
        for (int i = 0; i < n-1; i++) {
            dx[i] = new_kz[i+1] - new_kz[i];
        }
        
        // Calculate relative distance (error metric)
        std::vector<double> rel_distance(n-1, 0.0);
        
        // Add contributions from cubic interpolation errors
        for (int i = 0; i < n-2; i++) {
            double diff1 = std::abs(a1[i]) / dy2[i];
            double diff2 = std::abs(a2[i]) / dy2[i];
            
            rel_distance[i] += diff1;      // Add to interval [i, i+1]
            rel_distance[i+1] += diff2;    // Add to interval [i+1, i+2]
        }
        
        // Find intervals that are out of tolerance
        std::vector<int> to_split;
        for (int i = 0; i < n-1; i++) {
            if (rel_distance[i] > tolerance) {
                to_split.push_back(i);
            }
        }
        
        out_of_tolerance = to_split.size();
        
        if (out_of_tolerance > 0) {
            // Calculate new kz values (midpoints of intervals to split)
            std::vector<double> new_kz_vals(out_of_tolerance);
            std::vector<double> new_dR(out_of_tolerance);
            
            for (int i = 0; i < out_of_tolerance; i++) {
                int idx = to_split[i];
                new_kz_vals[i] = new_kz[idx] + dx[idx] / 2.0;
                new_dR[i] = mapped_dR[idx];
            }
            
            // Calculate reflectivity amplitudes at new kz points
            std::vector<Cplx> new_r(out_of_tolerance);
            std::vector<int> new_rho_index(out_of_tolerance, 0);
            
            reflectivity_amplitude(layers, depth, sigma, rho, irho,
                                  out_of_tolerance, &new_kz_vals[0], 
                                  &new_rho_index[0], &new_r[0]);
            
            // Calculate new R values
            std::vector<double> new_R(out_of_tolerance);
            for (int i = 0; i < out_of_tolerance; i++) {
                new_R[i] = std::norm(new_r[i]);
            }
            
            // Insert new values into arrays
            new_kz = insert_values(new_kz, to_split, new_kz_vals);
            r = insert_values(r, to_split, new_r);
            R = insert_values(R, to_split, new_R);
            mapped_dR = insert_values(mapped_dR, to_split, new_dR);
        }
        
        iterations++;
    }
    
    // Return the vectors as a pair
    return std::make_pair(new_kz, r);
}

// Backward-compatible wrapper for oversample_inplace
extern "C" void
oversample_inplace_legacy(const int initial_points,
                          const double initial_kz[],
                          const double initial_dR[],
                          const double tolerance,
                          const int layers,
                          const double depth[],
                          const double sigma[],
                          const double rho[],
                          const double irho[],
                          const int max_sampling_iterations,
                          int* final_points,
                          double final_kz[],
                          Cplx final_r[],
                          double final_dR[])
{
    auto result = oversample_inplace(initial_points, initial_kz, initial_dR, tolerance,
                                   layers, depth, sigma, rho, irho, max_sampling_iterations);
    
    *final_points = result.first.size();
    for (int i = 0; i < *final_points; i++) {
        final_kz[i] = result.first[i];
        final_r[i] = result.second[i];
        if (final_dR != nullptr) {
            // For legacy compatibility, fill dR with default values
            final_dR[i] = 0.01; // Default dR value
        }
    }
}

// Main autosampled reflectivity amplitude function that returns vectors
std::pair<std::vector<double>, std::vector<Cplx>>
autosampled_reflectivity_amplitude(const int layers,
                                   const double depth[],
                                   const double sigma[], 
                                   const double rho[],
                                   const double irho[],
                                   const int initial_points,
                                   const double initial_kz[],
                                   const int rho_index[], // Not used in this version
                                   const double initial_dR[],
                                   const double dQ[],     // Not used in this version
                                   const double tolerance,
                                   const int max_sampling_iterations)
{
    // Handle default dR if null
    std::vector<double> dR_values(initial_points);
    if (initial_dR != nullptr) {
        for (int i = 0; i < initial_points; i++) {
            dR_values[i] = initial_dR[i];
        }
    } else {
        // Default to 1% relative error
        for (int i = 0; i < initial_points; i++) {
            dR_values[i] = 0.01;
        }
    }
    
    // Call the oversampling function (returns vectors directly)
    return oversample_inplace(initial_points, initial_kz, &dR_values[0], tolerance,
                            layers, depth, sigma, rho, irho, max_sampling_iterations);
}

// Backward-compatible wrapper for autosampled_reflectivity_amplitude
extern "C" void
autosampled_reflectivity_amplitude_legacy(const int layers,
                                          const double depth[],
                                          const double sigma[], 
                                          const double rho[],
                                          const double irho[],
                                          const int initial_points,
                                          const double initial_kz[],
                                          const int rho_index[],
                                          const double initial_dR[],
                                          const double dQ[],
                                          const double tolerance,
                                          const int max_sampling_iterations,
                                          int* final_points,
                                          double final_kz[],
                                          Cplx final_r[])
{
    auto result = autosampled_reflectivity_amplitude(layers, depth, sigma, rho, irho,
                                                   initial_points, initial_kz, rho_index,
                                                   initial_dR, dQ, tolerance, max_sampling_iterations);
    
    *final_points = result.first.size();
    for (int i = 0; i < *final_points; i++) {
        final_kz[i] = result.first[i];
        final_r[i] = result.second[i];
    }
}