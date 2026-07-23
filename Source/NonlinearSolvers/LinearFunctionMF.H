/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef LinearFunctionMF_H_
#define LinearFunctionMF_H_

#include "LinearFunction.H"
#include "Utils/TextMsg.H"

#include <AMReX_Config.H>

/**
 * \brief This is a linear function class for computing the action of a
 *  linear operator on a vector using a matrix-free method.
 *  This class has all of the required functions to be used as the
 *  linear operator template parameter in AMReX_GMRES.
 */

template <class T, class Ops>
class LinearFunctionMF : public LinearFunction<T,Ops>
{
    public:

    using RT = typename T::value_type;

    LinearFunctionMF() = default;
    ~LinearFunctionMF() override = default;

    // Default move and copy operations
    LinearFunctionMF(const LinearFunctionMF&) = default;
    LinearFunctionMF& operator=(const LinearFunctionMF&) = default;
    LinearFunctionMF(LinearFunctionMF&&) noexcept = default;
    LinearFunctionMF& operator=(LinearFunctionMF&&) noexcept = default;

    void apply ( T& a_dF, const T& a_dU ) override;

    inline
    void precond ( T& a_U, const T& a_X ) override
    {
        a_U.Copy(a_X);
    }

    inline
    void updatePreCondMat ( const T& a_X ) override
    {
        amrex::ignore_unused(a_X);
    }

    inline
    void getPCMatrix( amrex::Gpu::DeviceVector<int>& a_ridx_g,
                      amrex::Gpu::DeviceVector<int>& a_nnz,
                      amrex::Gpu::DeviceVector<int>& a_cidx_g,
                      amrex::Gpu::DeviceVector<RT>&  a_aij,
                      int& a_n, int& a_ncols_max ) override
    {
        amrex::ignore_unused(a_ridx_g, a_nnz, a_cidx_g, a_aij, a_n, a_ncols_max);
    }

    T makeVecLHS () const override;
    T makeVecRHS () const override;

    [[nodiscard]] inline
    bool isDefined() const { return m_is_defined;  }

    void define( const T&, Ops*, const PreconditionerType& ) override;

    [[nodiscard]] inline
    PreconditionerType pcType () const override { return PreconditionerType::none; }

    private:

    bool m_is_defined = false;
    RT m_cur_time = RT(0.0);

    T m_R;
    Ops* m_ops = nullptr;
};

template <class T, class Ops>
void LinearFunctionMF<T,Ops>::define ( const T& a_U,
                                         Ops* a_ops,
                                         const PreconditionerType& a_pc_type )
{
    BL_PROFILE("LinearFunctionMF::::define()");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_pc_type == PreconditionerType::none,
        "LinearFunctionMF::define(): preconditioners are not supported");

    m_R.Define(a_U);
    m_ops = a_ops;
    m_is_defined = true;
}

template <class T, class Ops>
auto LinearFunctionMF<T,Ops>::makeVecRHS () const -> T
{
    BL_PROFILE("LinearFunctionMF::::makeVecRHS()");
    T x;
    x.Define(m_R);
    return x;
}

template <class T, class Ops>
auto LinearFunctionMF<T,Ops>::makeVecLHS () const -> T
{
    BL_PROFILE("LinearFunctionMF::::makeVecLHS()");
    T x;
    x.Define(m_R);
    return x;
}

template <class T, class Ops>
void LinearFunctionMF<T,Ops>::apply (T& a_dF, const T& a_dU)
{
    BL_PROFILE("LinearFunctionMF::apply()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        isDefined(),
        "LinearFunctionMF::apply() called on undefined LinearFunctionMF");

    m_ops->ComputeRHS(a_dF, a_dU, m_cur_time, -1, true );
}

#endif
