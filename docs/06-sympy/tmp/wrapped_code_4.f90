!******************************************************************************
!*                      Code generated with sympy 0.7.6                       *
!*                                                                            *
!*              See http://www.sympy.org/ for more information.               *
!*                                                                            *
!*                      This file is part of 'autowrap'                       *
!******************************************************************************

subroutine autofunc(a, b, c, out_1451769269)
implicit none
REAL*8, intent(in) :: a
REAL*8, intent(in) :: b
REAL*8, intent(in) :: c
REAL*8, intent(out), dimension(1:2, 1:1) :: out_1451769269

out_1451769269(1, 1) = (1.0d0/2.0d0)*(-b + sqrt(-4*a*c + b**2))/a
out_1451769269(2, 1) = -1.0d0/2.0d0*(b + sqrt(-4*a*c + b**2))/a

end subroutine
