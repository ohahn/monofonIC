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

#include <array>

#include <general.hh>
#include <grid_fft.hh>

template <typename data_t, typename derived_t>
class BaseConvolver
{
protected:
    std::array<size_t, 3> np_;
    std::array<real_t, 3> length_;

public:
    BaseConvolver(const std::array<size_t, 3> &N, const std::array<real_t, 3> &L)
        : np_(N), length_(L) {}

    BaseConvolver( const BaseConvolver& ) = delete;
    
    virtual ~BaseConvolver() {}

    // implements convolution of two Fourier-space fields
    template <typename kfunc1, typename kfunc2, typename opp>
    void convolve2(kfunc1 kf1, kfunc2 kf2, opp op) {}

    // implements convolution of three Fourier-space fields
    template <typename kfunc1, typename kfunc2, typename kfunc3, typename opp>
    void convolve3(kfunc1 kf1, kfunc2 kf2, kfunc3 kf3, opp op) {}

public:

    template <typename opp>
    void convolve_Gradients(Grid_FFT<data_t> &inl, const std::array<int, 1> &d1l,
                            Grid_FFT<data_t> &inr, const std::array<int, 1> &d1r,
                            opp output_op)
    {
        // transform to FS in case fields are not
        inl.FourierTransformForward();
        inr.FourierTransformForward();
        // perform convolution of Hessians
        static_cast<derived_t &>(*this).convolve2(
            [&inl,&d1l](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto grad1 = inl.gradient(d1l[0],{i,j,k});
                return grad1*inl.kelem(i, j, k);
            },
            [&inr,&d1r](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto grad1 = inr.gradient(d1r[0],{i,j,k});
                return grad1*inr.kelem(i, j, k);
            },
            output_op);
    }

    template <typename opp>
    void convolve_Gradient_and_Hessian(Grid_FFT<data_t> &inl, const std::array<int, 1> &d1l,
                                       Grid_FFT<data_t> &inr, const std::array<int, 2> &d2r,
                                       opp output_op)
    {
        // transform to FS in case fields are not
        inl.FourierTransformForward();
        inr.FourierTransformForward();
        // perform convolution of Hessians
        static_cast<derived_t &>(*this).convolve2(
            [&](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto kk = inl.template get_k<real_t>(i, j, k);
                return ccomplex_t(0.0, -kk[d1l[0]]) * inl.kelem(i, j, k);
            },
            [&](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto kk = inr.template get_k<real_t>(i, j, k);
                return -kk[d2r[0]] * kk[d2r[1]] * inr.kelem(i, j, k);
            },
            output_op);
    }

    template <typename opp>
    void convolve_Hessians(Grid_FFT<data_t> &inl, const std::array<int, 2> &d2l,
                           Grid_FFT<data_t> &inr, const std::array<int, 2> &d2r,
                           opp output_op)
    {
        // transform to FS in case fields are not
        inl.FourierTransformForward();
        inr.FourierTransformForward();
        // perform convolution of Hessians
        static_cast<derived_t &>(*this).convolve2(
            [&inl,&d2l](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto grad1 = inl.gradient(d2l[0],{i,j,k});
                auto grad2 = inl.gradient(d2l[1],{i,j,k});
                return grad1*grad2*inl.kelem(i, j, k);
            },
            [&inr,&d2r](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto grad1 = inr.gradient(d2r[0],{i,j,k});
                auto grad2 = inr.gradient(d2r[1],{i,j,k});
                return grad1*grad2*inr.kelem(i, j, k);
            },
            output_op);
    }

    template <typename opp>
    void convolve_Hessians(Grid_FFT<data_t> &inl, const std::array<int, 2> &d2l,
                           Grid_FFT<data_t> &inm, const std::array<int, 2> &d2m,
                           Grid_FFT<data_t> &inr, const std::array<int, 2> &d2r,
                           opp output_op)
    {
        // transform to FS in case fields are not
        inl.FourierTransformForward();
        inm.FourierTransformForward();
        inr.FourierTransformForward();
        // perform convolution of Hessians
        static_cast<derived_t &>(*this).convolve3(
            [&inl, &d2l](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto grad1 = inl.gradient(d2l[0],{i,j,k});
                auto grad2 = inl.gradient(d2l[1],{i,j,k});
                return grad1*grad2*inl.kelem(i, j, k);
            },
            [&inm, &d2m](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto grad1 = inm.gradient(d2m[0],{i,j,k});
                auto grad2 = inm.gradient(d2m[1],{i,j,k});
                return grad1*grad2*inm.kelem(i, j, k);
            },
            [&inr, &d2r](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto grad1 = inr.gradient(d2r[0],{i,j,k});
                auto grad2 = inr.gradient(d2r[1],{i,j,k});
                return grad1*grad2*inr.kelem(i, j, k);
            },
            output_op);
    }

    template <typename opp>
    void convolve_SumOfHessians(Grid_FFT<data_t> &inl, const std::array<int, 2> &d2l,
                                Grid_FFT<data_t> &inr, const std::array<int, 2> &d2r1, const std::array<int, 2> &d2r2,
                                opp output_op)
    {
        // transform to FS in case fields are not
        inl.FourierTransformForward();
        inr.FourierTransformForward();
        // perform convolution of Hessians
        static_cast<derived_t &>(*this).convolve2(
            [&inl, &d2l](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto grad1 = inl.gradient(d2l[0],{i,j,k});
                auto grad2 = inl.gradient(d2l[1],{i,j,k});
                return grad1*grad2*inl.kelem(i, j, k);
            },
            [&inr, &d2r1, &d2r2](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto grad11 = inr.gradient(d2r1[0],{i,j,k});
                auto grad12 = inr.gradient(d2r1[1],{i,j,k});
                auto grad21 = inr.gradient(d2r2[0],{i,j,k});
                auto grad22 = inr.gradient(d2r2[1],{i,j,k});
                return (grad11*grad12+grad21*grad22)*inr.kelem(i, j, k);
            },
            output_op);
    }

    template <typename opp>
    void convolve_DifferenceOfHessians(Grid_FFT<data_t> &inl, const std::array<int, 2> &d2l,
                                       Grid_FFT<data_t> &inr, const std::array<int, 2> &d2r1, const std::array<int, 2> &d2r2,
                                       opp output_op)
    {
        // transform to FS in case fields are not
        inl.FourierTransformForward();
        inr.FourierTransformForward();
        // perform convolution of Hessians
        static_cast<derived_t &>(*this).convolve2(
            [&inl, &d2l](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto grad1 = inl.gradient(d2l[0],{i,j,k});
                auto grad2 = inl.gradient(d2l[1],{i,j,k});
                return grad1*grad2*inl.kelem(i, j, k);
            },
            [&inr, &d2r1, &d2r2](size_t i, size_t j, size_t k) -> ccomplex_t {
                auto grad11 = inr.gradient(d2r1[0],{i,j,k});
                auto grad12 = inr.gradient(d2r1[1],{i,j,k});
                auto grad21 = inr.gradient(d2r2[0],{i,j,k});
                auto grad22 = inr.gradient(d2r2[1],{i,j,k});
                return (grad11*grad12-grad21*grad22)*inr.kelem(i, j, k);
            },
            output_op);
    }
};

//! naive convolution class, disrespecting aliasing
template <typename data_t>
class NaiveConvolver : public BaseConvolver<data_t, NaiveConvolver<data_t>>
{
protected:
    Grid_FFT<data_t> *fbuf1_, *fbuf2_;

    using BaseConvolver<data_t, NaiveConvolver<data_t>>::np_;
    using BaseConvolver<data_t, NaiveConvolver<data_t>>::length_;

public:
    NaiveConvolver(const std::array<size_t, 3> &N, const std::array<real_t, 3> &L)
        : BaseConvolver<data_t, NaiveConvolver<data_t>>(N, L)
    {
        fbuf1_ = new Grid_FFT<data_t>(N, length_, true, kspace_id);
        fbuf2_ = new Grid_FFT<data_t>(N, length_, true, kspace_id);
    }

    ~NaiveConvolver()
    {
        delete fbuf1_;
        delete fbuf2_;
    }

    template <typename kfunc1, typename kfunc2, typename opp>
    void convolve2(kfunc1 kf1, kfunc2 kf2, opp output_op)
    {
        //... prepare data 1
        fbuf1_->FourierTransformForward(false);
        this->copy_in(kf1, *fbuf1_);

        //... prepare data 2
        fbuf2_->FourierTransformForward(false);
        this->copy_in(kf2, *fbuf2_);

        //... convolve
        fbuf1_->FourierTransformBackward();
        fbuf2_->FourierTransformBackward();

#pragma omp parallel for
        for (size_t i = 0; i < fbuf1_->ntot_; ++i)
        {
            (*fbuf2_).relem(i) *= (*fbuf1_).relem(i);
        }
        fbuf2_->FourierTransformForward();
        // fbuf2_->dealias();
//... copy data back
#pragma omp parallel for
        for (size_t i = 0; i < fbuf2_->ntot_; ++i)
        {
            output_op(i, (*fbuf2_)[i]);
        }


    }

    template <typename kfunc1, typename kfunc2, typename kfunc3, typename opp>
    void convolve3(kfunc1 kf1, kfunc2 kf2, kfunc3 kf3, opp output_op)
    {
        //... prepare data 1
        fbuf1_->FourierTransformForward(false);
        this->copy_in(kf1, *fbuf1_);

        //... prepare data 2
        fbuf2_->FourierTransformForward(false);
        this->copy_in(kf2, *fbuf2_);

        //... convolve
        fbuf1_->FourierTransformBackward();
        fbuf2_->FourierTransformBackward();

#pragma omp parallel for
        for (size_t i = 0; i < fbuf1_->ntot_; ++i)
        {
            (*fbuf2_).relem(i) *= (*fbuf1_).relem(i);
        }

        //... prepare data 2
        fbuf1_->FourierTransformForward(false);
        this->copy_in(kf3, *fbuf1_);

        //... convolve
        fbuf1_->FourierTransformBackward();

#pragma omp parallel for
        for (size_t i = 0; i < fbuf1_->ntot_; ++i)
        {
            (*fbuf2_).relem(i) *= (*fbuf1_).relem(i);
        }

        fbuf2_->FourierTransformForward();
//... copy data back
#pragma omp parallel for
        for (size_t i = 0; i < fbuf2_->ntot_; ++i)
        {
            output_op(i, (*fbuf2_)[i]);
        }
    }

private:
    template <typename kfunc>
    void copy_in(kfunc kf, Grid_FFT<data_t> &g)
    {
#pragma omp parallel for
        for (size_t i = 0; i < g.size(0); ++i)
        {
            for (size_t j = 0; j < g.size(1); ++j)
            {
                for (size_t k = 0; k < g.size(2); ++k)
                {
                    g.kelem(i, j, k) = kf(i, j, k);
                }
            }
        }
    }
};

//! convolution class, respecting Orszag's 3/2 rule
template <typename data_t>
class OrszagConvolver : public BaseConvolver<data_t, OrszagConvolver<data_t>>
{
private:
    Grid_FFT<data_t> *f1p_, *f2p_;
    Grid_FFT<data_t> *fbuf_;

    using BaseConvolver<data_t, OrszagConvolver<data_t>>::np_;
    using BaseConvolver<data_t, OrszagConvolver<data_t>>::length_;

    ccomplex_t *crecvbuf_;
    real_t *recvbuf_;
    size_t maxslicesz_;
    std::vector<ptrdiff_t> offsets_, offsetsp_;
    std::vector<size_t> sizes_, sizesp_;

    int get_task(ptrdiff_t index, const std::vector<ptrdiff_t> &offsets, const std::vector<size_t> &sizes, const int ntasks)
    {
        int itask = 0;
        while (itask < ntasks - 1 && offsets[itask + 1] <= index)
            ++itask;
        return itask;
    }

public:
    OrszagConvolver(const std::array<size_t, 3> &N, const std::array<real_t, 3> &L)
        : BaseConvolver<data_t, OrszagConvolver<data_t>>({3 * N[0] / 2, 3 * N[1] / 2, 3 * N[2] / 2}, L)
    {
        //... create temporaries
        f1p_ = new Grid_FFT<data_t>(np_, length_, true, kspace_id);
        f2p_ = new Grid_FFT<data_t>(np_, length_, true, kspace_id);
        fbuf_ = new Grid_FFT<data_t>(N, length_, true, kspace_id); // needed for MPI, or for triple conv.

#if defined(USE_MPI)
        maxslicesz_ = f1p_->sizes_[1] * f1p_->sizes_[3] * 2;

        crecvbuf_ = new ccomplex_t[maxslicesz_ / 2];
        recvbuf_ = reinterpret_cast<real_t *>(&crecvbuf_[0]);

        int ntasks(MPI::get_size());

        offsets_.assign(ntasks, 0);
        offsetsp_.assign(ntasks, 0);
        sizes_.assign(ntasks, 0);
        sizesp_.assign(ntasks, 0);

        size_t tsize = N[0], tsizep = f1p_->size(0);

        MPI_Allgather(&fbuf_->local_1_start_, 1, MPI_LONG_LONG, &offsets_[0], 1,
                      MPI_LONG_LONG, MPI_COMM_WORLD);
        MPI_Allgather(&f1p_->local_1_start_, 1, MPI_LONG_LONG, &offsetsp_[0], 1,
                      MPI_LONG_LONG, MPI_COMM_WORLD);
        MPI_Allgather(&tsize, 1, MPI_LONG_LONG, &sizes_[0], 1, MPI_LONG_LONG,
                      MPI_COMM_WORLD);
        MPI_Allgather(&tsizep, 1, MPI_LONG_LONG, &sizesp_[0], 1, MPI_LONG_LONG,
                      MPI_COMM_WORLD);
#endif
    }

    ~OrszagConvolver()
    {
        delete f1p_;
        delete f2p_;
        delete fbuf_;
#if defined(USE_MPI)
        delete[] crecvbuf_;
#endif
    }

    template <typename kfunc1, typename kfunc2, typename opp>
    void convolve2(kfunc1 kf1, kfunc2 kf2, opp output_op)
    {
        //... prepare data 1
        f1p_->FourierTransformForward(false);
        this->pad_insert(kf1, *f1p_);

        //... prepare data 2
        f2p_->FourierTransformForward(false);
        this->pad_insert(kf2, *f2p_);

        //... convolve
        f1p_->FourierTransformBackward();
        f2p_->FourierTransformBackward();

#pragma omp parallel for
        for (size_t i = 0; i < f1p_->ntot_; ++i)
        {
            (*f2p_).relem(i) *= (*f1p_).relem(i);
        }
        f2p_->FourierTransformForward();
        //... copy data back
        unpad(*f2p_, output_op);
    }

    template <typename kfunc1, typename kfunc2, typename kfunc3, typename opp>
    void convolve3(kfunc1 kf1, kfunc2 kf2, kfunc3 kf3, opp output_op)
    {
        auto assign_to = [](auto &g) { return [&](auto i, auto v) { g[i] = v; }; };
        fbuf_->FourierTransformForward(false);
        convolve2(kf1, kf2, assign_to(*fbuf_));
        convolve2([&](size_t i, size_t j, size_t k) -> ccomplex_t { return fbuf_->kelem(i, j, k); }, kf3, output_op);
    }

    // template< typename opp >
    // void test_pad_unpad( Grid_FFT<data_t> & in, Grid_FFT<data_t> & res, opp op )
    // {
    //     //... prepare data 1
    //     f1p_->FourierTransformForward(false);
    //     this->pad_insert( [&in]( size_t i, size_t j, size_t k ){return in.kelem(i,j,k);}, *f1p_ );
    //     f1p_->FourierTransformBackward();
    //     f1p_->FourierTransformForward();
    //     res.FourierTransformForward();
    //     unpad(*f1p_, res, op);
    // }

private:

    template <typename kdep_functor>
    void pad_insert( kdep_functor kfunc, Grid_FFT<data_t> &fp)
    {
        const real_t rfac = std::pow(1.5, 1.5);

#if !defined(USE_MPI)
        const size_t nhalf[3] = {fp.n_[0] / 3, fp.n_[1] / 3, fp.n_[2] / 3};

        fp.zero();

        #pragma omp parallel for
        for (size_t i = 0; i < 2 * fp.size(0) / 3; ++i)
        {
            size_t ip = (i > nhalf[0]) ? i + nhalf[0] : i;
            for (size_t j = 0; j < 2 * fp.size(1) / 3; ++j)
            {
                size_t jp = (j > nhalf[1]) ? j + nhalf[1] : j;
                for (size_t k = 0; k < nhalf[2]+1; ++k)
                {
                    size_t kp = (k > nhalf[2]) ? k + nhalf[2] : k;
                    fp.kelem(ip, jp, kp) = kfunc(i, j, k) * rfac;
                }
            }
        }
#else
        fbuf_->FourierTransformForward(false);
        
        #pragma omp parallel for
        for (size_t i = 0; i < fbuf_->size(0); ++i)
        {
            for (size_t j = 0; j < fbuf_->size(1); ++j)
            {
                for (size_t k = 0; k < fbuf_->size(2); ++k)
                {
                    fbuf_->kelem(i, j, k) = kfunc(i, j, k) * rfac;
                }
            }
        }

        fbuf_->FourierInterpolateCopyTo( fp );
        
#endif //defined(USE_MPI)
    }

    template <typename operator_t>
    void unpad( Grid_FFT<data_t> &fp, operator_t output_op)
    {
        const real_t rfac = std::sqrt(fp.n_[0] * fp.n_[1] * fp.n_[2]) / std::sqrt(fbuf_->n_[0] * fbuf_->n_[1] * fbuf_->n_[2]);

        // make sure we're in Fourier space...
        assert(fp.space_ == kspace_id);

#if !defined(USE_MPI) ////////////////////////////////////////////////////////////////////////////////////
        fbuf_->FourierTransformForward(false);
        size_t nhalf[3] = {fbuf_->n_[0] / 2, fbuf_->n_[1] / 2, fbuf_->n_[2] / 2};

        #pragma omp parallel for
        for (size_t i = 0; i < fbuf_->size(0); ++i)
        {
            size_t ip = (i > nhalf[0]) ? i + nhalf[0] : i;
            for (size_t j = 0; j < fbuf_->size(1); ++j)
            {
                size_t jp = (j > nhalf[1]) ? j + nhalf[1] : j;
                for (size_t k = 0; k < fbuf_->size(2); ++k)
                {
                    size_t kp = (k > nhalf[2]) ? k + nhalf[2] : k;
                    fbuf_->kelem(i, j, k) = fp.kelem(ip, jp, kp) / rfac;
                    // zero Nyquist modes since they are not unique after convolution
                    if( i==nhalf[0]||j==nhalf[1]||k==nhalf[2]){
                        fbuf_->kelem(i, j, k) = 0.0; 
                    }
                }
            }
        }

        //... copy data back
        #pragma omp parallel for
        for (size_t i = 0; i < fbuf_->ntot_; ++i)
        {
            output_op(i, (*fbuf_)[i]);
        }

#else /// then USE_MPI is defined //////////////////////////////////////////////////////////////
    
        fp.FourierInterpolateCopyTo( *fbuf_ );

        //... copy data back
        #pragma omp parallel for
        for (size_t i = 0; i < fbuf_->ntot_; ++i)
        {

            output_op(i, (*fbuf_)[i] / rfac);
        }

#endif //defined(USE_MPI)
    }
};
