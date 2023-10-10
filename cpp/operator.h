// Copyright (c) 2016-2023 Ivan Iakoupov
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#ifndef OPERATOR_H
#define OPERATOR_H

#include <cstdint>
#include <vector>
#include <unordered_map>

#include "types.h"
#include "lookup3.h"

typedef uint16_t sigma_state_t;

class BasisVector
{
    std::vector<sigma_state_t> m_spec;
    double m_factor;
public:
    BasisVector() : m_factor(0) {}
    explicit BasisVector(const std::vector<sigma_state_t> &spec, double factor = 1) :
        m_spec(spec),
        m_factor(factor)
    {}
    bool isAtomInState(int j, sigma_state_t state)
    {
        return m_spec.at(j) == state;
    }
    sigma_state_t &operator[](int j)
    {
        return m_spec[j];
    }
    const sigma_state_t &operator[](int j) const
    {
        return m_spec[j];
    }
    std::size_t size() const { return m_spec.size(); }
    std::vector<sigma_state_t> spec() const { return m_spec; }
    void setSpec(std::vector<sigma_state_t> &spec, double factor = 1) {
        m_spec = spec;
        m_factor = factor;
    }
    void setFactor(double factor)
    {
        m_factor = factor;
    }
    double factor() const { return m_factor; }
};

bool operator==(const BasisVector &a, const BasisVector &b);

namespace std
{
template<>
struct hash<BasisVector>
{
    size_t operator()(const BasisVector &vec) const
    {
        return hashlittle(vec.spec().data(), vec.spec().size(), 12345);
    }
};
} // namespace std

enum HarmonicOscillatorOperatorType
{
    SigmaOperatorType,
    CreationOperatorType,
    AnnihilationOperatorType
};

class Sigma
{
    sigma_state_t m_dst;
    sigma_state_t m_src;
    unsigned int m_j;
    HarmonicOscillatorOperatorType m_opType;
public:
    Sigma(sigma_state_t dst, sigma_state_t src, unsigned int j);
    Sigma(std::initializer_list<unsigned int> l);
    Sigma(unsigned int j, HarmonicOscillatorOperatorType opType);

    BasisVector sigmaActOn(const BasisVector &vec) const;
    BasisVector creationOpActOn(const BasisVector &vec) const;
    BasisVector annihilationOpActOn(const BasisVector &vec) const;
    BasisVector operator()(const BasisVector &vec) const;
    std::vector<Eigen::Triplet<std::complex<double>>>
    triplets(const std::vector<BasisVector> &basis) const;
    std::vector<Eigen::Triplet<std::complex<double>>>
    triplets(const std::vector<BasisVector> &basis,
             const std::unordered_map<BasisVector, int> &basis_map) const;
    std::vector<Eigen::Triplet<double>>
    tripletsReal(const std::vector<BasisVector> &basis) const;
    std::vector<Eigen::Triplet<double>>
    tripletsReal(const std::vector<BasisVector> &basis,
             const std::unordered_map<BasisVector, int> &basis_map) const;
    SpMat matrix(const std::vector<BasisVector> &basis) const;
    SpMat matrix(const std::vector<BasisVector> &basis,
                 const std::unordered_map<BasisVector, int> &basis_map) const;
    SpMatReal matrixReal(const std::vector<BasisVector> &basis) const;
    SpMatReal matrixReal(const std::vector<BasisVector> &basis,
                 const std::unordered_map<BasisVector, int> &basis_map) const;
};

class SigmaProduct
{
    // The factors are stored in the reverse order,
    // so that when the whole product needs to be
    // applied onto a state, this array needs
    // to be iterated in the usual forward manner.
    std::vector<Sigma> m_sigma_factors;
public:
    SigmaProduct() = default;
    SigmaProduct(std::initializer_list<Sigma> l);
    void addFactorInFront(Sigma s);
    BasisVector actOn(const BasisVector &vec) const;
    BasisVector operator()(const BasisVector &vec) const;

    std::vector<Eigen::Triplet<std::complex<double>>>
    triplets(const std::vector<BasisVector> &basis) const;
    std::vector<Eigen::Triplet<std::complex<double>>>
    triplets(const std::vector<BasisVector> &basis,
             const std::unordered_map<BasisVector, int> &basis_map) const;
    std::vector<Eigen::Triplet<double>>
    tripletsReal(const std::vector<BasisVector> &basis) const;
    std::vector<Eigen::Triplet<double>>
    tripletsReal(const std::vector<BasisVector> &basis,
             const std::unordered_map<BasisVector, int> &basis_map) const;
    SpMat matrix(const std::vector<BasisVector> &basis) const;
    SpMat matrix(const std::vector<BasisVector> &basis,
                 const std::unordered_map<BasisVector, int> &basis_map) const;
    SpMatReal matrixReal(const std::vector<BasisVector> &basis) const;
    SpMatReal matrixReal(const std::vector<BasisVector> &basis,
                 const std::unordered_map<BasisVector, int> &basis_map) const;
};

std::string stringify_spec(const std::vector <sigma_state_t> &spec);
std::vector<BasisVector> generate_basis_vectors(int NAtoms, int NExcitations);
std::vector<BasisVector> generate_all_basis_vectors(int NAtoms);
std::vector<BasisVector> generate_tensor_product_basis(
        const std::vector<int> &subsystem_excitations);
std::vector<BasisVector> generate_basis_vectors_harmonic(int NOscillators,
                                                         int NExcitations);
std::vector<BasisVector> generate_basis_vectors_combined_two_level_and_harmonic(
        int NAtoms, int NOscillators, int NExcitations);
std::vector<BasisVector> generate_basis_vectors_combined_transmon_and_harmonic(
        const std::vector<int> &transmon_excitations, int NOscillators,
        int NExcitations);
std::unordered_map<BasisVector, int> generate_basis_map(
        const std::vector<BasisVector> &basis);

#endif // OPERATOR_H

