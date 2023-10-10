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

#include "operator.h"

#define FACTOR_TOLERANCE 1e-16

Sigma::Sigma(sigma_state_t dst, sigma_state_t src, unsigned int j) :
    m_dst(dst), m_src(src), m_j(j), m_opType(SigmaOperatorType)
{
}

Sigma::Sigma(std::initializer_list<unsigned int> l) {
    assert(l.size() == 3 && "Exactly 3 arguments are needed to define sigma!");
    const unsigned int *ptr = l.begin();
    assert(*ptr <= std::numeric_limits<sigma_state_t>::max()+1
           && "Integer overflow detected!");
    m_dst = static_cast<sigma_state_t>(*ptr);
    ++ptr;
    assert(*ptr <= std::numeric_limits<sigma_state_t>::max()+1
           && "Integer overflow detected!");
    m_src = static_cast<sigma_state_t>(*ptr);
    ++ptr;
    m_j = *ptr;
    m_opType = SigmaOperatorType;
}

Sigma::Sigma(unsigned int j, HarmonicOscillatorOperatorType opType) :
    m_dst(0), m_src(0), m_j(j), m_opType(opType)
{
    assert(m_opType != SigmaOperatorType
           && "Use another constructor for SigmaOperatorType");
}

BasisVector Sigma::sigmaActOn(const BasisVector &vec) const
{
    assert(m_j < vec.size() && "The basis vector is too short!");
    BasisVector ret = vec;
    if (ret[m_j] == m_src) {
        ret[m_j] = m_dst;
    } else {
        ret.setFactor(0);
    }
    return ret;
}

BasisVector Sigma::creationOpActOn(const BasisVector &vec) const
{
    assert(m_j < vec.size() && "The basis vector is too short!");
    BasisVector ret = vec;
    ret.setFactor(std::sqrt(ret[m_j]+1)*ret.factor());
    ++ret[m_j];
    return ret;
}

BasisVector Sigma::annihilationOpActOn(const BasisVector &vec) const
{
    assert(m_j < vec.size() && "The basis vector is too short!");
    BasisVector ret = vec;
    ret.setFactor(std::sqrt(ret[m_j])*ret.factor());
    --ret[m_j];
    return ret;
}

BasisVector Sigma::operator()(const BasisVector &vec) const
{
    switch (m_opType) {
    case SigmaOperatorType:
        return sigmaActOn(vec);
    case CreationOperatorType:
        return creationOpActOn(vec);
    case AnnihilationOperatorType:
        return annihilationOpActOn(vec);
    default:
        assert(0 && "Unknown operator type!");
        return BasisVector();
    }
}

std::vector<Eigen::Triplet<std::complex<double>>>
Sigma::triplets(const std::vector<BasisVector> &basis) const
{
    SigmaProduct prod = {*this};
    return prod.triplets(basis);
}

std::vector<Eigen::Triplet<std::complex<double>>>
Sigma::triplets(const std::vector<BasisVector> &basis,
                const std::unordered_map<BasisVector, int> &basis_map) const
{
    SigmaProduct prod = {*this};
    return prod.triplets(basis, basis_map);
}

std::vector<Eigen::Triplet<double>>
Sigma::tripletsReal(const std::vector<BasisVector> &basis) const
{
    SigmaProduct prod = {*this};
    return prod.tripletsReal(basis);
}

std::vector<Eigen::Triplet<double>>
Sigma::tripletsReal(const std::vector<BasisVector> &basis,
                const std::unordered_map<BasisVector, int> &basis_map) const
{
    SigmaProduct prod = {*this};
    return prod.tripletsReal(basis, basis_map);
}

SpMat Sigma::matrix(const std::vector<BasisVector> &basis) const
{
    SigmaProduct prod = {*this};
    return prod.matrix(basis);
}

SpMat Sigma::matrix(const std::vector<BasisVector> &basis,
                    const std::unordered_map<BasisVector, int> &basis_map) const
{
    SigmaProduct prod = {*this};
    return prod.matrix(basis, basis_map);
}

SpMatReal Sigma::matrixReal(const std::vector<BasisVector> &basis) const
{
    SigmaProduct prod = {*this};
    return prod.matrixReal(basis);
}

SpMatReal Sigma::matrixReal(const std::vector<BasisVector> &basis,
                    const std::unordered_map<BasisVector, int> &basis_map) const
{
    SigmaProduct prod = {*this};
    return prod.matrixReal(basis, basis_map);
}

bool operator==(const BasisVector &a, const BasisVector &b)
{
    const int aSize = a.size();
    const int bSize = b.size();
    if (aSize != bSize) {
        return false;
    }
    for (int i = 0; i < aSize; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

SigmaProduct::SigmaProduct(std::initializer_list<Sigma> l)
{
    // Place the elements of the initializer
    // list in the reverse order into the member
    // array.
    const int l_size = l.size();
    m_sigma_factors.reserve(l_size);
    const Sigma *ptr = l.end()-1;
    for (int i = 0; i < l_size; ++i) {
        m_sigma_factors.push_back(*ptr);
        --ptr;
    }
}

void SigmaProduct::addFactorInFront(Sigma s)
{
    m_sigma_factors.push_back(s);
}

BasisVector SigmaProduct::actOn(const BasisVector &vec) const
{
    BasisVector ret = vec;
    for (const Sigma &s: m_sigma_factors) {
        const BasisVector s_ret = s(ret);
        ret = s_ret;
    }
    return ret;
}

BasisVector SigmaProduct::operator()(const BasisVector &vec) const
{
    return actOn(vec);
}

std::vector<Eigen::Triplet<std::complex<double>>>
SigmaProduct::triplets(const std::vector<BasisVector> &basis) const
{
    int basisSize = basis.size();
    std::vector<Eigen::Triplet<std::complex<double>>> ret;
    for (int m = 0; m < basisSize; ++m) {
        BasisVector transformed = actOn(basis[m]);
        if (std::abs(transformed.factor()) > FACTOR_TOLERANCE) {
            for (int n = 0; n < basisSize; ++n) {
                if (transformed == basis[n]) {
                    ret.emplace_back(n, m, transformed.factor());
                }
            }
        }
    }
    return ret;
}

std::vector<Eigen::Triplet<std::complex<double>>>
SigmaProduct::triplets(const std::vector<BasisVector> &basis,
                       const std::unordered_map<BasisVector, int> &basis_map) const
{
    int basisSize = basis.size();
    std::vector<Eigen::Triplet<std::complex<double>>> ret;
    for (int m = 0; m < basisSize; ++m) {
        BasisVector transformed = actOn(basis[m]);
        if (std::abs(transformed.factor()) > FACTOR_TOLERANCE) {
            int n = basis_map.at(transformed);
            ret.emplace_back(n, m, transformed.factor());
        }
    }
    return ret;
}

std::vector<Eigen::Triplet<double>>
SigmaProduct::tripletsReal(const std::vector<BasisVector> &basis) const
{
    int basisSize = basis.size();
    std::vector<Eigen::Triplet<double>> ret;
    for (int m = 0; m < basisSize; ++m) {
        BasisVector transformed = actOn(basis[m]);
        if (std::abs(transformed.factor()) > FACTOR_TOLERANCE) {
            for (int n = 0; n < basisSize; ++n) {
                if (transformed == basis[n]) {
                    ret.emplace_back(n, m, transformed.factor());
                }
            }
        }
    }
    return ret;
}

std::vector<Eigen::Triplet<double>> SigmaProduct::tripletsReal(
        const std::vector<BasisVector> &basis,
        const std::unordered_map<BasisVector, int> &basis_map) const
{
    int basisSize = basis.size();
    std::vector<Eigen::Triplet<double>> ret;
    for (int m = 0; m < basisSize; ++m) {
        BasisVector transformed = actOn(basis[m]);
        if (std::abs(transformed.factor()) > FACTOR_TOLERANCE) {
            int n = basis_map.at(transformed);
            ret.emplace_back(n, m, transformed.factor());
        }
    }
    return ret;
}

SpMat SigmaProduct::matrix(const std::vector<BasisVector> &basis) const
{
    std::vector<Eigen::Triplet<std::complex<double>>> t = triplets(basis);
    SpMat M(basis.size(), basis.size());
    M.setFromTriplets(t.begin(), t.end());
    return M;
}

SpMat SigmaProduct::matrix(const std::vector<BasisVector> &basis,
                           const std::unordered_map<BasisVector, int> &basis_map) const
{
    std::vector<Eigen::Triplet<std::complex<double>>> t = triplets(basis,
                                                                   basis_map);
    SpMat M(basis.size(), basis.size());
    M.setFromTriplets(t.begin(), t.end());
    return M;
}

SpMatReal SigmaProduct::matrixReal(const std::vector<BasisVector> &basis) const
{
    std::vector<Eigen::Triplet<double>> t = tripletsReal(basis);
    SpMatReal M(basis.size(), basis.size());
    M.setFromTriplets(t.begin(), t.end());
    return M;
}

SpMatReal SigmaProduct::matrixReal(const std::vector<BasisVector> &basis,
                           const std::unordered_map<BasisVector, int> &basis_map) const
{
    std::vector<Eigen::Triplet<double>> t = tripletsReal(basis, basis_map);
    SpMatReal M(basis.size(), basis.size());
    M.setFromTriplets(t.begin(), t.end());
    return M;
}

std::string stringify_spec(const std::vector<sigma_state_t> &spec)
{
    int size = spec.size();
    if (size == 0) {
        return std::string("");
    }
    std::stringstream stream;
    stream << static_cast<int>(spec[size-1]);
    for(int i = size-2; i >= 0; --i) {
        stream << " " << static_cast<int>(spec[i]);
    }
    return stream.str();
}

std::vector<BasisVector> generate_basis_vectors(int NAtoms, int NExcitations)
{
    assert(NExcitations <= NAtoms
           && "Cannot put more excitations than two-level atoms!");
    std::vector<BasisVector> ret;
    std::vector<sigma_state_t> spec(NAtoms, 0);
    for (int i = 0; i < NExcitations; ++i) {
        spec[i] = 1;
    }
    do {
        BasisVector vec(spec);
        ret.push_back(vec);
    } while (std::prev_permutation(spec.begin(),spec.end()));
    return ret;
}

std::vector<BasisVector> generate_all_basis_vectors(int NAtoms)
{
    std::vector<BasisVector> basis;
    basis.reserve(std::ldexp(1, NAtoms));
    for (int i = 0; i <= NAtoms; ++i) {
        std::vector<BasisVector> basis_i = generate_basis_vectors(NAtoms, i);
        basis.insert(basis.end(), basis_i.cbegin(), basis_i.cend());
    }
    return basis;
}

std::vector<BasisVector> generate_tensor_product_basis(
        const std::vector<int> &subsystem_excitations)
{
    const int NSubsystems = subsystem_excitations.size();
    std::vector<BasisVector> basis;
    if (NSubsystems == 0) {
        return basis;
    }
    const int subsystem_excitations_0 = subsystem_excitations[0];
    for (int k = 0; k <= subsystem_excitations_0; ++k) {
        std::vector<sigma_state_t> spec(NSubsystems, 0);
        spec[0] = k;
        basis.emplace_back(spec);
    }
    std::vector<BasisVector> basis_new;
    for (int i = 1; i < NSubsystems; ++i) {
        const int previous_basis_size = basis.size();
        for (int j = 0; j < previous_basis_size; ++j) {
            const int subsystem_excitations_i = subsystem_excitations[i];
            for (int k = 0; k <= subsystem_excitations_i; ++k) {
                std::vector<sigma_state_t> spec = basis[j].spec();
                spec[i] = k;
                basis_new.emplace_back(spec);
            }
        }
        basis = basis_new;
        basis_new.clear();
    }
    return basis;
}

std::vector<int> spec_to_index_array(const std::vector<sigma_state_t> &spec)
{
    std::vector<int> ret;
    for (int i = 0; i < spec.size(); ++i) {
        if (spec[i] != 0) {
            ret.push_back(i);
        }
    }
    return ret;
}

std::vector<BasisVector> generate_basis_vectors_harmonic(int NOscillators, int NExcitations)
{
    assert(NExcitations <= std::numeric_limits<sigma_state_t>::max()+1
           && "Integer overflow detected!");
    std::vector<BasisVector> ret;
    if (NExcitations == 0) {
        std::vector<sigma_state_t> spec_to_add(NOscillators, 0);
        BasisVector vec(spec_to_add);
        ret.push_back(vec);
        return ret;
    }
    const int upper_limit = std::min(NOscillators, NExcitations);
    for (int j = 1; j <= upper_limit; ++j) {
        // This spec determines which harmonic oscillators
        // are "active", i.e., have non-zero excitations.
        std::vector<sigma_state_t> spec_active(NOscillators, 0);

        const int j_clamped = std::min(NOscillators, j);
        for (int i = 0; i < j_clamped; ++i) {
            spec_active[i] = 1;
        }
        do {
            std::vector<int> active_indices = spec_to_index_array(spec_active);
            const int num_separators = std::max(0, j_clamped-1);
            const int spec_separator_size = NExcitations-1;
            std::vector<sigma_state_t> spec_separator(spec_separator_size, 0);
            for (int i = 0; i < num_separators; ++i) {
                spec_separator[i] = 1;
            }
            do {
                std::vector<sigma_state_t> spec_to_add(NOscillators, 0);
                int index = 0;
                // We have the convention that the value "1" means
                // a separator to the right of this position. Hence,
                // the implicit separator at the beginning could be
                // thought as an explicit separator to the right of
                // the index = -1.
                int last_i = -1;
                for (int i = 0; i < spec_separator_size; ++i) {
                    if (spec_separator[i] == 1) {
                        spec_to_add[active_indices[index]] = i-last_i;
                        last_i = i;
                        ++index;
                    }
                }

                // Add the last block value
                spec_to_add[active_indices[index]] = NExcitations-1-last_i;

                BasisVector vec(spec_to_add);
                ret.push_back(vec);
            } while (std::prev_permutation(spec_separator.begin(),spec_separator.end()));
        } while (std::prev_permutation(spec_active.begin(),spec_active.end()));
    }
    return ret;
}

std::vector<BasisVector> generate_basis_vectors_combined_two_level_and_harmonic(
        int NAtoms, int NOscillators, int NExcitations)
{
    std::vector<BasisVector> ret;
    // We cannot put more excitations in the two-level atoms
    // than the number of those two-level atoms
    const int NExcitationsTwoLevel = std::min(NAtoms, NExcitations);
    for (int i = 0; i <= NExcitationsTwoLevel; ++i) {
        const std::vector<BasisVector> two_level_basis
                = generate_basis_vectors(NAtoms, i);
        const std::vector<BasisVector> harmonic_basis
                = generate_basis_vectors_harmonic(NOscillators, NExcitations-i);
        const int two_level_basis_size = two_level_basis.size();
        const int harmonic_basis_size = harmonic_basis.size();
        const int basis_vector_size = NAtoms + NOscillators;
        for (int j = 0; j < two_level_basis_size; ++j) {
            for (int k = 0; k < harmonic_basis_size; ++k) {
                std::vector<sigma_state_t> spec(basis_vector_size, 0);
                for (int m = 0; m < NAtoms; ++m) {
                    spec[m] = two_level_basis[j][m];
                }
                for (int m = 0; m < NOscillators; ++m) {
                    spec[m+NAtoms] = harmonic_basis[k][m];
                }
                BasisVector vec(spec);
                ret.push_back(vec);
            }
        }
    }
    return ret;
}

std::vector<BasisVector> generate_basis_vectors_combined_transmon_and_harmonic(
        const std::vector<int> &transmon_excitations, int NOscillators, int NExcitations)
{
    std::vector<BasisVector> ret;
    const int NAtoms = transmon_excitations.size();
    const int basis_vector_size = NAtoms + NOscillators;
    // What we do below is to generate all transmon basis vectors
    // and then discard the ones that have more than "NExcitations"
    // TODO: this is inefficient, and what we really want is to
    // be able to generate only the transmon basis vectors that
    // have desired number of excitations, like it is possible for
    // the two-level atoms with "generate_basis_vectors".
    const std::vector<BasisVector> transmon_basis
            = generate_tensor_product_basis(transmon_excitations);
    const int transmon_basis_size = transmon_basis.size();
    for (int i = 0; i < transmon_basis_size; ++i) {
        int NExcitationsTransmon = 0;
        for (int j = 0; j < NAtoms; ++j) {
            NExcitationsTransmon += transmon_basis[i][j];
        }
        if (NExcitationsTransmon > NExcitations) {
            continue;
        }
        const std::vector<BasisVector> harmonic_basis
                = generate_basis_vectors_harmonic(
                        NOscillators, NExcitations-NExcitationsTransmon);
        const int harmonic_basis_size = harmonic_basis.size();
        for (int j = 0; j < harmonic_basis_size; ++j) {
            std::vector<sigma_state_t> spec(basis_vector_size, 0);
            for (int m = 0; m < NAtoms; ++m) {
                spec[m] = transmon_basis[i][m];
            }
            for (int m = 0; m < NOscillators; ++m) {
                spec[m+NAtoms] = harmonic_basis[j][m];
            }
            ret.emplace_back(spec);
        }
    }
    return ret;
}

std::unordered_map<BasisVector, int> generate_basis_map(const std::vector<BasisVector> &basis)
{
    std::unordered_map<BasisVector, int> basis_map;
    const int basis_size = basis.size();
    basis_map.reserve(basis_size);
    for (int i = 0; i < basis_size; ++i) {
        basis_map[basis[i]] = i;
    }
    return basis_map;
}
