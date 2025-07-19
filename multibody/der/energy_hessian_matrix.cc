#include "drake/multibody/der/energy_hessian_matrix.h"

#include <algorithm>

#include "drake/multibody/der/constraint_participation.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

template <typename T>
EnergyHessianMatrix<T> EnergyHessianMatrix<T>::Allocate(int num_dofs) {
  DRAKE_THROW_UNLESS(num_dofs >= 7);
  DRAKE_THROW_UNLESS(num_dofs % 4 == 0 || num_dofs % 4 == 3);
  const bool has_closed_ends = (num_dofs % 4 == 0);
  const int num_nodes = (num_dofs + 1) / 4;

  std::vector<int> block_sizes(num_nodes, 4);
  std::vector<std::vector<int>> neighbors(num_nodes);
  if (!has_closed_ends) {
    /*
     ⎡ ██ ██ ██             ⎤
     ⎢ ██ ██ ██ ██          ⎥
     ⎢ ██ ██ ██ ██ ██       ⎥
     ⎢    ██ ██ ██ ██ ██    ⎥
     ⎢       ██ ██ ██ ██ ██ ⎥
     ⎢          ██ ██ ██ ██ ⎥
     ⎣             ██ ██ ██ ⎦
    */
    for (int i = 0; i < num_nodes; ++i) {
      neighbors[i].push_back(i);
      if (i + 1 < num_nodes) neighbors[i].push_back(i + 1);
      if (i + 2 < num_nodes) neighbors[i].push_back(i + 2);
    }
  } else {
    /*
     ⎡ ██ ██ ██       ██ ██ ⎤
     ⎢ ██ ██ ██ ██       ██ ⎥
     ⎢ ██ ██ ██ ██ ██       ⎥
     ⎢    ██ ██ ██ ██ ██    ⎥
     ⎢       ██ ██ ██ ██ ██ ⎥
     ⎢ ██       ██ ██ ██ ██ ⎥
     ⎣ ██ ██       ██ ██ ██ ⎦
    */
    for (int i = 0; i < num_nodes; ++i) {
      neighbors[i].push_back(i);
      if (i + 1 < num_nodes)
        neighbors[i].push_back(i + 1);
      else
        neighbors[(i + 1) % num_nodes].push_back(i);
      if (i + 2 < num_nodes)
        neighbors[i].push_back(i + 2);
      else
        neighbors[(i + 2) % num_nodes].push_back(i);
    }
  }

  contact_solvers::internal::BlockSparsityPattern block_sparsity_pattern(
      std::move(block_sizes), std::move(neighbors));
  return EnergyHessianMatrix<T>(
      num_dofs, contact_solvers::internal::Block4x4SparseSymmetricMatrix<T>(
                    std::move(block_sparsity_pattern)));
}

template <typename T>
EnergyHessianMatrixVectorProduct<T> EnergyHessianMatrix<T>::operator*(
    const Eigen::VectorX<T>& vec) const {
  DRAKE_THROW_UNLESS(vec.size() == cols());
  return EnergyHessianMatrixVectorProduct<T>(this, &vec);
}

template <typename T>
void EnergyHessianMatrix<T>::SetZero() {
  data_.SetZero();
}

template <typename T>
void EnergyHessianMatrix<T>::Insert(DerNodeIndex i, DerNodeIndex j,
                                    const Eigen::Ref<const Matrix3<T>>& mat) {
  Eigen::Matrix4<T> filler = Eigen::Matrix4<T>::Zero();
  filler.template topLeftCorner<3, 3>() = mat;
  if (int{i} >= int{j})
    data_.AddToBlock(i, j, filler);
  else
    data_.AddToBlock(j, i, filler.transpose());
}

template <typename T>
void EnergyHessianMatrix<T>::Insert(DerNodeIndex i, DerEdgeIndex j,
                                    const Eigen::Ref<const Vector3<T>>& vec) {
  Eigen::Matrix4<T> filler = Eigen::Matrix4<T>::Zero();
  filler.template topRightCorner<3, 1>() = vec;
  if (int{i} == int{j}) {
    data_.AddToBlock(i, j, filler + filler.transpose());
  } else {
    if (int{i} > int{j})
      data_.AddToBlock(i, j, filler);
    else
      data_.AddToBlock(j, i, filler.transpose());
  }
}

template <typename T>
void EnergyHessianMatrix<T>::Insert(DerEdgeIndex i, DerEdgeIndex j,
                                    const T& val) {
  Eigen::Matrix4<T> filler = Eigen::Matrix4<T>::Zero();
  filler(3, 3) = val;
  if (int{i} >= int{j})
    data_.AddToBlock(i, j, filler);
  else
    data_.AddToBlock(j, i, filler.transpose());
}

template <typename T>
void EnergyHessianMatrix<T>::AddScaledMatrix(
    const Eigen::DiagonalMatrix<T, Eigen::Dynamic>& rhs, const T& scale) {
  DRAKE_THROW_UNLESS(rhs.rows() == rows());
  if (scale == 0.0) return;

  for (int i = 0; i < data_.block_rows(); ++i) {
    if (data_.rows() == rows() || i < data_.block_rows() - 1) {
      auto vec = rhs.diagonal().template segment<4>(4 * i) * scale;
      data_.AddToBlock(i, i, vec.asDiagonal().toDenseMatrix());
    } else {
      Eigen::Vector4<T> vec = Eigen::Vector4<T>::Zero();
      vec.template head<3>() =
          rhs.diagonal().template segment<3>(4 * i) * scale;
      data_.AddToBlock(i, i, vec.asDiagonal().toDenseMatrix());
    }
  }
}

template <typename T>
void EnergyHessianMatrix<T>::AddScaledMatrix(const EnergyHessianMatrix& rhs,
                                             const T& scale) {
  DRAKE_THROW_UNLESS(rhs.rows() == rows());
  if (scale == 0.0) return;

  for (int j = 0; j < rhs.data_.block_cols(); ++j) {
    for (int i : rhs.data_.sparsity_pattern().neighbors()[j]) {  // i ≥ j
      data_.AddToBlock(i, j, rhs.data_.block(i, j) * scale);
    }
  }
}

template <typename T>
void EnergyHessianMatrix<T>::ApplyBoundaryCondition(DerNodeIndex node_index) {
  for (int j = 0; j < data_.block_cols(); ++j) {
    for (int i : data_.sparsity_pattern().neighbors()[j]) {  // i ≥ j
      if (!(i == node_index || j == node_index)) continue;
      Eigen::Matrix4<T> block = data_.block(i, j);
      if (i == node_index) block.template topRows<3>().setZero();
      if (j == node_index) block.template leftCols<3>().setZero();
      if (i == j) block.template topLeftCorner<3, 3>().setIdentity();
      data_.SetBlock(i, j, block);
    }
  }
}

template <typename T>
void EnergyHessianMatrix<T>::ApplyBoundaryCondition(DerEdgeIndex edge_index) {
  for (int j = 0; j < data_.block_cols(); ++j) {
    for (int i : data_.sparsity_pattern().neighbors()[j]) {  // i ≥ j
      if (!(i == edge_index || j == edge_index)) continue;
      Eigen::Matrix4<T> block = data_.block(i, j);
      if (i == edge_index) block.template bottomRows<1>().setZero();
      if (j == edge_index) block.template rightCols<1>().setZero();
      if (i == j) block(3, 3) = 1.0;
      data_.SetBlock(i, j, block);
    }
  }
}

template <typename T>
Eigen::SparseMatrix<T> EnergyHessianMatrix<T>::ComputeLowerTriangle() const {
  std::vector<int> reserve_sizes(cols());
  reserve_sizes.reserve(cols());
  for (int block_j = 0; block_j < data_.block_cols(); ++block_j) {
    for (int v = 0; v < 4; ++v) {
      const int j = block_j * 4 + v;
      if (j < cols()) {
        reserve_sizes[j] =
            data_.sparsity_pattern().neighbors()[block_j].size() * 4;
      }
    }
  }

  Eigen::SparseMatrix<T> result(rows(), cols());
  result.reserve(reserve_sizes);
  for (int block_j = 0; block_j < data_.block_cols(); ++block_j) {
    for (int v = 0; v < 4; ++v) {
      const int j = 4 * block_j + v;
      if (j >= cols()) continue;
      for (int block_i : data_.sparsity_pattern().neighbors()[block_j]) {
        // block_i ≥ block_j
        const Matrix4<T>& block = data_.block(block_i, block_j);
        for (int u = 0; u < 4; ++u) {
          const int i = 4 * block_i + u;
          if (i >= rows()) continue;
          result.insert(i, j) = block(u, v);
        }
      }
    }
  }
  return result;
}

template <typename T>
static void FillInTriplets(
    int block_i, int block_j, const Eigen::Ref<const Matrix4<T>>& block,
    const contact_solvers::internal::PartialPermutation& permutation,
    int A_size, std::vector<Eigen::Triplet<T>>* A_triplets,
    std::vector<Eigen::Triplet<T>>* Bt_triplets,
    std::vector<Eigen::Triplet<T>>* D_triplets) {
  DRAKE_THROW_UNLESS(permutation.permuted_domain_size() ==
                     permutation.domain_size());
  const int num_dofs = permutation.domain_size();
  constexpr int block_size = 4;
  for (int i = 0; i < block_size; ++i) {
    for (int j = 0; j < block_size; ++j) {
      const int dof_i = block_size * block_i + i;
      const int dof_j = block_size * block_j + j;
      if (dof_i >= num_dofs || dof_j >= num_dofs) continue;
      const int permuted_dof_i = permutation.permutation()[dof_i];
      const int permuted_dof_j = permutation.permutation()[dof_j];
      if (permuted_dof_i < A_size && permuted_dof_j < A_size) {
        A_triplets->emplace_back(permuted_dof_i, permuted_dof_j, block(i, j));
      } else if (permuted_dof_i >= A_size && permuted_dof_j >= A_size) {
        D_triplets->emplace_back(permuted_dof_i - A_size,
                                 permuted_dof_j - A_size, block(i, j));
      } else if (permuted_dof_i >= A_size && permuted_dof_j < A_size) {
        Bt_triplets->emplace_back(permuted_dof_i - A_size, permuted_dof_j,
                                  block(i, j));
      } else {  // Do nothing since we don't need to fill in B.
      }
    }
  }
}

template <typename T>
template <typename T1>
std::enable_if_t<std::is_same_v<T1, double>, SchurComplement<T>>
EnergyHessianMatrix<T>::ComputeSchurComplement(
    const std::unordered_set<int>& participating_dofs) const {
  const int num_dofs = num_dofs_;
  const int num_participating_dofs = participating_dofs.size();
  contact_solvers::internal::PartialPermutation permutation =
      ComputeDofPermutation(num_dofs, participating_dofs);
  permutation.ExtendToFullPermutation();

  std::vector<Eigen::Triplet<T>> A_triplets;
  std::vector<Eigen::Triplet<T>> Bt_triplets;
  std::vector<Eigen::Triplet<T>> D_triplets;

  for (int block_j = 0; block_j < data_.block_cols(); ++block_j) {
    for (int block_i :
         data_.sparsity_pattern().neighbors()[block_j]) {  // block_i ≥ block_j
      const Matrix4<T>& block = data_.block(block_i, block_j);
      FillInTriplets<T>(block_i, block_j, block, permutation,
                        num_participating_dofs, &A_triplets, &Bt_triplets,
                        &D_triplets);
      if (block_i == block_j) continue;
      FillInTriplets<T>(block_j, block_i, block.transpose(), permutation,
                        num_participating_dofs, &A_triplets, &Bt_triplets,
                        &D_triplets);
    }
  }

  Eigen::SparseMatrix<T> A(num_participating_dofs, num_participating_dofs);
  A.setFromTriplets(A_triplets.begin(), A_triplets.end());
  Eigen::SparseMatrix<T> Bt(num_dofs - num_participating_dofs,
                            num_participating_dofs);
  Bt.setFromTriplets(Bt_triplets.begin(), Bt_triplets.end());
  Eigen::SparseMatrix<T> D(num_dofs - num_participating_dofs,
                           num_dofs - num_participating_dofs);
  D.setFromTriplets(D_triplets.begin(), D_triplets.end());
  return SchurComplement<T>(A, Bt, D);
}

template <typename T>
Eigen::MatrixX<T> EnergyHessianMatrix<T>::MakeDenseMatrix() const {
  const int size = (num_dofs_ + 1) / 4 * 4;
  Eigen::MatrixX<T> result = Eigen::MatrixX<T>::Zero(size, size);

  for (int j = 0; j < data_.block_cols(); ++j) {
    for (int i : data_.sparsity_pattern().neighbors()[j]) {  // i ≥ j
      const Matrix4<T>& block = data_.block(i, j);
      result.template block<4, 4>(4 * i, 4 * j) = block;
      if (i == j) continue;
      result.template block<4, 4>(4 * j, 4 * i) = block.transpose();
    }
  }
  return result.topLeftCorner(num_dofs_, num_dofs_);
}

template <typename T>
EnergyHessianMatrix<T>::EnergyHessianMatrix(
    int num_dofs,
    contact_solvers::internal::Block4x4SparseSymmetricMatrix<T>&& data)
    : num_dofs_(num_dofs), data_(std::move(data)) {
  DRAKE_THROW_UNLESS(num_dofs >= 7);
  DRAKE_THROW_UNLESS(num_dofs % 4 == 0 || num_dofs % 4 == 3);
}

template <typename T>
Eigen::Ref<Eigen::VectorX<T>> EnergyHessianMatrixVectorProduct<T>::AddToVector(
    EigenPtr<Eigen::VectorX<T>> lhs) const {
  DRAKE_THROW_UNLESS(lhs->size() == vec_->size());
  const auto& matdata = mat_->data_;

  for (int j = 0; j < matdata.block_cols(); ++j) {
    for (int i : matdata.sparsity_pattern().neighbors()[j]) {  // i ≥ j
      if (matdata.rows() == matdata.rows() || i < matdata.block_rows() - 1) {
        if (i == j) {
          lhs->template segment<4>(4 * i) +=
              matdata.block(i, j) * vec_->template segment<4>(4 * j) * scale_;
        } else {
          lhs->template segment<4>(4 * i) +=
              matdata.block(i, j) * vec_->template segment<4>(4 * j) * scale_;
          lhs->template segment<4>(4 * j) += matdata.block(i, j).transpose() *
                                             vec_->template segment<4>(4 * i) *
                                             scale_;
        }
      } else {
        if (i == j) {
          lhs->template segment<3>(4 * i) +=
              matdata.block(i, j).template topLeftCorner<3, 3>() *
              vec_->template segment<3>(4 * j) * scale_;
        } else {
          lhs->template segment<3>(4 * i) +=
              matdata.block(i, j).template topLeftCorner<3, 4>() *
              vec_->template segment<4>(4 * j) * scale_;
          lhs->template segment<4>(4 * j) +=
              matdata.block(i, j).template topLeftCorner<3, 4>().transpose() *
              vec_->template segment<3>(4 * i) * scale_;
        }
      }
    }
  }
  return *lhs;
}

template SchurComplement<double>
EnergyHessianMatrix<double>::ComputeSchurComplement<double>(
    const std::unordered_set<int>&) const;

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::internal::EnergyHessianMatrix);

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::internal::EnergyHessianMatrixVectorProduct);
