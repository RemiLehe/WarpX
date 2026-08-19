/* Copyright 2025 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef LinearFunction_H_
#define LinearFunction_H_

#include "Preconditioner.H"

#include <AMReX_BLProfiler.H>

#include <string>

/**
 * \brief This is a base class for defining a linear function class
 *        to use with WarpX implicit solvers.
 */

template <class T, class Ops>
class LinearFunction
{
    public:

        using RT = typename T::value_type;

        //! default constructor
        LinearFunction() = default;
        //! default destructore
        virtual ~LinearFunction() = default;

        // Default move and copy operations
        LinearFunction(const LinearFunction&) = default;
        LinearFunction& operator=(const LinearFunction&) = default;
        LinearFunction(LinearFunction&&) noexcept = default;
        LinearFunction& operator=(LinearFunction&&) noexcept = default;

        /**
         * \brief Apply the linear operator A represented by this
         *        LinearFunction object on a given vector of type T,
         *        i.e., compute its action a_Ax = A(a_x).
         *
         * \param a_Ax result of applying the operator, i.e., A(a_x).
         * \param a_x  vector the operator is applied to.
         */
        virtual void apply ( T& a_Ax, const T& a_x ) = 0;

        /**
         * \brief Apply the preconditioner on a given vector of type T,
         *        i.e., solve P(a_U) = a_X for a_U, where P is an
         *        approximation of the linear operator A represented by
         *        this LinearFunction object (so a_U is an approximation
         *        of the solution of A(a_U) = a_X). If there is no
         *        preconditioner, P = I and a_U is a copy of a_X.
         *
         * \param a_U result of applying the preconditioner, i.e., P^{-1}(a_X).
         * \param a_X vector the preconditioner is applied to.
         */
        virtual void precond ( T& a_U, const T& a_X ) = 0;

        //! update preconditioner
        virtual void updatePreCondMat ( const T&  a_X ) = 0;

        //! get sparse matrix representation of preconditioner
        virtual void getPCMatrix( amrex::Gpu::DeviceVector<int>&,
                                  amrex::Gpu::DeviceVector<int>&,
                                  amrex::Gpu::DeviceVector<int>&,
                                  amrex::Gpu::DeviceVector<RT>&,
                                  int&, int& ) = 0;

        //! create a new vector given a defined vector
        inline void create ( T& a_Z, const T& a_U )
        {
            a_Z.define(a_U);
        }

        /**
         * \brief Make a vector suitable to be used as the LHS x in A(x) = b.
         *
         * The default implementation returns a vector with the same layout as
         * vecTemplate(). Override this only if the operator needs the LHS to
         * have a different layout than the RHS (for instance, extra ghost
         * cells for efficiency), which AMReX's GMRES allows for.
         */
        [[nodiscard]] virtual T makeVecLHS () const { return makeVec(); }

        /**
         * \brief Make a vector suitable to be used as the RHS b in A(x) = b.
         *        See makeVecLHS() for the LHS/RHS distinction.
         */
        [[nodiscard]] virtual T makeVecRHS () const { return makeVec(); }

        //! vector copy operation
        inline void assign( T& a_Z, const T& a_U ) {
            a_Z.Copy(a_U);
        }

        //! vector scaled-increment operation
        inline void increment( T& a_Z, const T& a_U, RT a_scale )
        {
            a_Z.increment(a_U,a_scale);
        }

        //! vector scaled operation
        inline void scale ( T& a_U, RT a_scale )
        {
            a_U.scale(a_scale);
        }

        //! compute linear combination
        inline void linComb ( T& a_U, RT a, const T& X, RT b, const T& Y )
        {
            a_U.linComb( a, X, b, Y );
        }

        //! set vector to zero
        inline void setToZero ( T& a_U )
        {
            a_U.zero();
        }

        //! set vector to a scalar value
        inline void setVal ( T& a_U, RT a_val )
        {
            a_U.setVal(a_val);
        }

        //! compute dot product of two vectors
        inline RT dotProduct( const T& a_X, const T& a_Y )
        {
            return( a_X.dotProduct(a_Y) );
        }

        //! compute 2-norm of a vector
        inline RT norm2( const T& a_U )
        {
            return ( a_U.norm2() );
        }

        //! define the linear function object
        virtual void define( const T&, Ops*, const PreconditionerType& ) = 0;

        //! returns the preconditioner type
        [[nodiscard]] virtual PreconditionerType pcType () const = 0;

    protected:

        /**
         * \brief Return a defined vector whose layout (grids, distribution
         *        map, number of components, ...) is that of the vectors this
         *        linear operator acts on. It is used by makeVecLHS() and
         *        makeVecRHS() to create new vectors.
         */
        [[nodiscard]] virtual const T& vecTemplate () const = 0;

        //! make a new, defined vector with the same layout as vecTemplate()
        [[nodiscard]] T makeVec () const
        {
            BL_PROFILE("LinearFunction::makeVec()");
            T x;
            x.Define(vecTemplate());
            return x;
        }

    private:

};

#endif
