// This file is part of monofonIC (MUSIC2)
// A software package to generate ICs for cosmological simulations
// Copyright (C) 2020 by Oliver Hahn
// 
// monofonIC is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// monofonIC is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#pragma once

#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>

#include <math/vec3.hh>

/// @brief class for 3x3 matrix calculations
/// @tparam T type of matrix elements
template<typename T>
class mat3_t{
protected:
    std::array<T,9> data_; //< data array
    std::array<double,9> data_double_; //< data array for GSL operations
    gsl_matrix_view m_; //< GSL matrix view
    gsl_vector *eval_; //< GSL eigenvalue vector
    gsl_matrix *evec_; //< GSL eigenvector matrix
	gsl_eigen_symmv_workspace * wsp_; //< GSL workspace
    bool bdid_alloc_gsl_; //< flag to indicate whether GSL memory has been allocated
						
    /// @brief initialize GSL memory
    void init_gsl(){
        // allocate memory for GSL operations if we haven't done so yet
        if( !bdid_alloc_gsl_ )
        {
            if( typeid(T)!=typeid(double) ){
                m_ = gsl_matrix_view_array ( &data_double_[0], 3, 3);
            }else{
                m_ = gsl_matrix_view_array ( (double*)(&data_[0]), 3, 3); // this should only ever be called for T==double so cast is to avoid constexpr ifs from C++17
            }
            eval_ = gsl_vector_alloc (3);
            evec_ = gsl_matrix_alloc (3, 3);
            wsp_ = gsl_eigen_symmv_alloc (3);
            bdid_alloc_gsl_ = true;
        }

        if( typeid(T)!=typeid(double) ){
            for( int i=0; i<9; ++i ) data_double_[i] = double(data_[i]);
        }
    }

    /// @brief free GSL memory
    void free_gsl(){
        // free memory for GSL operations if it was allocated
        if( bdid_alloc_gsl_ )
        {
            gsl_eigen_symmv_free (wsp_);
            gsl_vector_free (eval_);
            gsl_matrix_free (evec_);
        }
    }

public:

    /// @brief default constructor
    mat3_t()
    : bdid_alloc_gsl_(false) 
    {}

    /// @brief copy constructor
    /// @param m matrix to copy
    mat3_t( const mat3_t<T> &m)
    : data_(m.data_), bdid_alloc_gsl_(false) 
    {}
    
    /// @brief move constructor
    /// @param m matrix to move
    mat3_t( mat3_t<T> &&m)
    : data_(std::move(m.data_)), bdid_alloc_gsl_(false) 
    {}

    /// @brief construct mat3_t from initializer list
    /// @param e initializer list
    template<typename ...E>
    mat3_t(E&&...e) 
    : data_{{std::forward<E>(e)...}}, bdid_alloc_gsl_(false)
    {}

    /// @brief assignment operator
    /// @param m matrix to copy
    /// @return reference to this
    mat3_t<T>& operator=(const mat3_t<T>& m) noexcept{
        data_ = m.data_;
        return *this;
    }

    /// @brief move assignment operator
    /// @param m matrix to move
    /// @return reference to this
    mat3_t<T>& operator=(const mat3_t<T>&& m) noexcept{
        data_ = std::move(m.data_);
        return *this;
    }

    /// @brief destructor
    ~mat3_t(){
        this->free_gsl();
    }
    
    /// @brief bracket index access to flattened matrix components
    /// @param i index
    /// @return reference to i-th component
    T &operator[](size_t i) noexcept { return data_[i];}
    
    /// @brief const bracket index access to flattened matrix components
    /// @param i index
    /// @return const reference to i-th component
    const T &operator[](size_t i) const noexcept { return data_[i]; }

    /// @brief matrix 2d index access
    /// @param i row index
    /// @param j column index
    /// @return reference to (i,j)-th component
    T &operator()(size_t i, size_t j) noexcept { return data_[3*i+j]; }

    /// @brief const matrix 2d index access
    /// @param i row index
    /// @param j column index
    /// @return const reference to (i,j)-th component
    const T &operator()(size_t i, size_t j) const noexcept { return data_[3*i+j]; }

    /// @brief in-place addition
    /// @param rhs matrix to add
    /// @return reference to this
    mat3_t<T>& operator+=( const mat3_t<T>& rhs ) noexcept{
        for (size_t i = 0; i < 9; ++i) {
           (*this)[i] += rhs[i];
        }
        return *this;
    }

    /// @brief in-place subtraction
    /// @param rhs matrix to subtract
    /// @return reference to this
    mat3_t<T>& operator-=( const mat3_t<T>& rhs ) noexcept{
        for (size_t i = 0; i < 9; ++i) {
           (*this)[i] -= rhs[i];
        }
        return *this;
    }

    /// @brief zeroing of matrix
    void zero() noexcept{
        for (size_t i = 0; i < 9; ++i) data_[i]=0;
    }

    /// @brief compute eigenvalues and eigenvectors
    /// @param evals eigenvalues
    /// @param evec1 first eigenvector
    /// @param evec2 second eigenvector
    /// @param evec3 third eigenvector
    void eigen( vec3_t<T>& evals, vec3_t<T>& evec1, vec3_t<T>& evec2, vec3_t<T>& evec3_t )
    {
        this->init_gsl();

        gsl_eigen_symmv (&m_.matrix, eval_, evec_, wsp_);
        gsl_eigen_symmv_sort (eval_, evec_, GSL_EIGEN_SORT_VAL_ASC);

        for( int i=0; i<3; ++i ){
            evals[i] = gsl_vector_get( eval_, i );
            evec1[i] = gsl_matrix_get( evec_, i, 0 );
            evec2[i] = gsl_matrix_get( evec_, i, 1 );
            evec3_t[i] = gsl_matrix_get( evec_, i, 2 );
        }
    }
};

/// @brief matrix addition
/// @tparam T type of matrix components
/// @param lhs left hand side matrix
/// @param rhs right hand side matrix
/// @return matrix result = lhs + rhs
template<typename T>
constexpr const mat3_t<T> operator+(const mat3_t<T> &lhs, const mat3_t<T> &rhs) noexcept
{
    mat3_t<T> result;
    for (size_t i = 0; i < 9; ++i) {
        result[i] = lhs[i] + rhs[i];
    }
    return result;
}

/// @brief matrix - vector multiplication
/// @tparam T type of matrix and vector components
/// @param A matrix
/// @param v vector
/// @return vector result = A*v
template<typename T>
inline vec3_t<T> operator*( const mat3_t<T> &A, const vec3_t<T> &v ) noexcept
{
    vec3_t<T> result;
    for( int mu=0; mu<3; ++mu ){
        result[mu] = 0.0;
        for( int nu=0; nu<3; ++nu ){
            result[mu] += A(mu,nu)*v[nu];
        }
    }
    return result;
}

