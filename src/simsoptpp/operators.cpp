#include <complex>

std::complex<double> operator+(const double lhs, const std::complex<double>& rhs) {
    return std::complex<double>(lhs) + rhs;  
}
std::complex<double> operator+(const std::complex<double>& lhs, const double rhs) {
    return lhs + std::complex<double>(rhs);  
}
std::complex<double> operator-(const double lhs, const std::complex<double>& rhs) {
    return std::complex<double>(lhs) - rhs; 
}
std::complex<double> operator-(const std::complex<double>& lhs, const double rhs) {
    return lhs - std::complex<double>(rhs); 
}
std::complex<double> operator*(const double lhs, const std::complex<double>& rhs) {
    return std::complex<double>(lhs) * rhs;  
}
std::complex<double> operator*(const std::complex<double>& lhs, const double rhs) {
    return lhs * std::complex<double>(rhs);  
}
std::complex<double> operator/(const double lhs, const std::complex<double>& rhs) {
    return std::complex<double>(lhs) / rhs;  
}
std::complex<double> operator/(const std::complex<double>& lhs, const double rhs) {
    return lhs / std::complex<double>(rhs);  
}
std::complex<double> operator+(const int lhs, const std::complex<double>& rhs) {
    return std::complex<double>(lhs) + rhs;  
}
std::complex<double> operator+(const std::complex<double>& lhs, const int rhs) {
    return lhs + std::complex<double>(rhs);  
}
std::complex<double> operator-(const int lhs, const std::complex<double>& rhs) {
    return std::complex<double>(lhs) - rhs; 
}
std::complex<double> operator-(const std::complex<double>& lhs,const  int rhs) {
    return lhs - std::complex<double>(rhs); 
}
std::complex<double> operator*(const int lhs, const std::complex<double>& rhs) {
    return std::complex<double>(lhs) * rhs;  
}
std::complex<double> operator*(const std::complex<double>& lhs, const int rhs) {
    return lhs * std::complex<double>(rhs);  
}
std::complex<double> operator/(const int lhs, const std::complex<double>& rhs) {
    return std::complex<double>(lhs) / rhs;  
}
std::complex<double> operator/(const std::complex<double>& lhs, const int rhs) {
    return lhs / std::complex<double>(rhs);  
}

