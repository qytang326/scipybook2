!******************************************************************************
!*                      Code generated with sympy 0.7.6                       *
!*                                                                            *
!*              See http://www.sympy.org/ for more information.               *
!*                                                                            *
!*                      This file is part of 'autowrap'                       *
!******************************************************************************

subroutine autofunc(m1, m2, l, g, q, theta, u, omega, out_948586642)
implicit none
REAL*8, intent(in) :: m1
REAL*8, intent(in) :: m2
REAL*8, intent(in) :: l
REAL*8, intent(in) :: g
REAL*8, intent(in) :: q
REAL*8, intent(in) :: theta
REAL*8, intent(in) :: u
REAL*8, intent(in) :: omega
REAL*8, intent(out), dimension(1:4, 1:1) :: out_948586642

out_948586642(1, 1) = u
out_948586642(2, 1) = omega
out_948586642(3, 1) = g*m2*sin(theta)*cos(theta)/(m1 + m2*sin(theta)**2 &
      ) + l*m2*omega**2*(m2*cos(theta)**2/((m1 + m2)*(m1 + m2*sin(theta &
      )**2)) + 1.0/(m1 + m2))*sin(theta)
out_948586642(4, 1) = -g*(m1 + m2)*sin(theta)/(l*(m1 + m2*sin(theta)**2 &
      )) - m2*omega**2*sin(theta)*cos(theta)/(m1 + m2*sin(theta)**2)

end subroutine
