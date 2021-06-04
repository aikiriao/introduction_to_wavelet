#!/bin/bash
gnuplot << EOF
    set terminal pdf enhanced color
    set terminal pdf size 8, 6 font ",24"
    set samples 1024
    set xlabel 't'
    set xtics 1
    set ylabel 'psi(t)'
    set ytics 1
    set grid
    haar_mother(x) = ((x >= 0) && (x < 0.5)) ? 1 : ((x >= 0.5) && (x < 1)) ? -1 : 0
    haar(x, a, b) = 1 / sqrt(a) * haar_mother((x - b) / a)
    haar_father(x) = ((x >= 0) && (x < 1)) ? 1 : 0
    haar_scaling(x, a, b) = 1 / sqrt(a) * haar_father((x - b) / a)
    haar_scaling_discrete(x, m, n) = haar_scaling(x, 2**(-m), n * 2**(-m))
    mexicanhat_mother(x) = (1 - 2 * (x ** 2)) * exp(-(x ** 2))
    mexicanhat(x, a, b) = 1 / sqrt(a) * mexicanhat_mother((x - b) / a)
    sinc(x) = sin(pi * x) / (pi * x)
    shanon_mother(x) = 2 * sinc(2 * x) - sinc(x)
    shanon(x,a,b) = 1 / sqrt(a) * shanon_mother((x - b) / a)

    set xrange [-5:5]
    set yrange [-1.5:1.5]
    set output "haar_wavelet.pdf"
    plot haar(x, 1, 0) t "Haar wavelet" linewidth 5 linetype rgb "red"
    set output "mexicanhat_wavelet.pdf"
    plot mexicanhat(x, 1, 0) t "Mexican hat wavelet" linewidth 5 linetype rgb "red"
    set output "shanon_wavelet.pdf"
    plot shanon(x, 1, 0) t "Shanon wavelet" linewidth 5 linetype rgb "red"

    set xrange [-1:10]
    set yrange [-2.1:2.1]
    set output "haar_wavelets.pdf"
    plot haar(x, 1, 0) t "a=1, b=0" linewidth 5 linetype rgb "black", \
        haar(x, 0.25, 3) t "a=0.25, b=3" linewidth 5 linetype rgb "red" dashtype '_', \
        haar(x, 4, 5) t "a=4, b=5" linewidth 5 linetype rgb "blue" dashtype '.'

    set ylabel 'phi(t)'
    set xrange [-1:10]
    set yrange [-0.1:2.1]
    set output "haar_scaling_functions.pdf"
    plot haar_scaling_discrete(x, 0, 0) t "m=0, n=0" linewidth 5 linetype rgb "black", \
        haar_scaling_discrete(x, 2, 8) t "m=2, n=8" linewidth 5 linetype rgb "red" dashtype '_', \
        haar_scaling_discrete(x, -2, 1) t "m=-2, n=1" linewidth 5 linetype rgb "blue" dashtype '.'

EOF
