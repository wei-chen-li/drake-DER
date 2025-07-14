#include "drake/multibody/contact_solvers/matrix_block.h"

#include "drake/common/overloaded.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

template <class T>
MatrixBlock<T>::MatrixBlock(Block3x3SparseMatrix<T> data)
    : data_(std::move(data)) {}

template <class T>
MatrixBlock<T>::MatrixBlock(Eigen::SparseMatrix<T, Eigen::RowMajor> data)
    : data_(std::move(data)) {}

template <class T>
MatrixBlock<T>::MatrixBlock(MatrixX<T> data) : data_(std::move(data)) {}

template <class T>
int MatrixBlock<T>::rows() const {
  return std::visit(
      [](auto&& arg) {
        /* We need the static_cast here because Eigen's rows() is `long`. */
        return static_cast<int>(arg.rows());
      },
      data_);
}

template <class T>
int MatrixBlock<T>::cols() const {
  return std::visit(
      [](auto&& arg) {
        /* We need the static_cast here because Eigen's cols() is `long`. */
        return static_cast<int>(arg.cols());
      },
      data_);
}

template <class T>
void MatrixBlock<T>::MultiplyAndAddTo(const Eigen::Ref<const MatrixX<T>>& A,
                                      EigenPtr<MatrixX<T>> y) const {
  DRAKE_THROW_UNLESS(y != nullptr);
  DRAKE_THROW_UNLESS(cols() == A.rows());
  DRAKE_THROW_UNLESS(rows() == y->rows());
  DRAKE_THROW_UNLESS(A.cols() == y->cols());
  std::visit(overloaded{
                 [&](const Block3x3SparseMatrix<T>& M) {
                   M.MultiplyAndAddTo(A, y);
                 },
                 [&](const auto& M) {
                   *y += M * A;
                 },
             },
             data_);
}

template <class T>
void MatrixBlock<T>::TransposeAndMultiplyAndAddTo(
    const Eigen::Ref<const MatrixX<T>>& A, EigenPtr<MatrixX<T>> y) const {
  DRAKE_THROW_UNLESS(y != nullptr);
  DRAKE_THROW_UNLESS(cols() == y->rows());
  DRAKE_THROW_UNLESS(rows() == A.rows());
  DRAKE_THROW_UNLESS(A.cols() == y->cols());
  std::visit(overloaded{
                 [&](const Block3x3SparseMatrix<T>& M) {
                   M.TransposeAndMultiplyAndAddTo(A, y);
                 },
                 [&](const auto& M) {
                   *y += M.transpose() * A;
                 },
             },
             data_);
}

template <class T>
void MatrixBlock<T>::TransposeAndMultiplyAndAddTo(
    const MatrixBlock<T>& A_block, EigenPtr<MatrixX<T>> y) const {
  DRAKE_THROW_UNLESS(y != nullptr);
  DRAKE_THROW_UNLESS(cols() == y->rows());
  DRAKE_THROW_UNLESS(rows() == A_block.rows());
  DRAKE_THROW_UNLESS(A_block.cols() == y->cols());
  std::visit(
      [&](const auto& M, const auto& A) {
        /* y += Mᵀ * A. */
        using MType = std::decay_t<decltype(M)>;
        using AType = std::decay_t<decltype(A)>;
        if constexpr (std::is_same_v<MType, Block3x3SparseMatrix<T>>) {
          M.TransposeAndMultiplyAndAddTo(A, y);
        } else if constexpr (std::is_same_v<AType, Block3x3SparseMatrix<T>>) {
          if constexpr (std::is_same_v<MType, MatrixX<T>>) {
            A.LeftMultiplyAndAddTo(M.transpose(), y);
          } else {
            // TODO(wei-chen): Utilize the sparseness here.
            A.LeftMultiplyAndAddTo(M.transpose().toDense(), y);
          }
        } else {
          *y += M.transpose() * A;
        }
      },
      this->data_, A_block.data_);
}

// TODO(xuchenhan-tri): consider a double dispatch strategy where each block
// type provides an API to operate on every other block type.
template <class T>
MatrixBlock<T> MatrixBlock<T>::LeftMultiplyByBlockDiagonal(
    const std::vector<MatrixX<T>>& Gs, int start, int end) const {
  DRAKE_THROW_UNLESS(start >= 0);
  DRAKE_THROW_UNLESS(end >= start);
  DRAKE_THROW_UNLESS(static_cast<int>(Gs.size()) > end);
  /* Verify that the sizes of G and M is compatible. */
  std::vector<int>
      row_starts;  // starting row index for each diagonal block in G
  row_starts.reserve(end - start + 1);
  int row = 0;
  for (int i = start; i <= end; ++i) {
    DRAKE_THROW_UNLESS(Gs[i].rows() == Gs[i].cols());
    row_starts.emplace_back(row);
    row += Gs[i].rows();
    if (std::holds_alternative<Block3x3SparseMatrix<T>>(data_)) {
      DRAKE_THROW_UNLESS(Gs[i].rows() % 3 == 0);
    }
  }
  DRAKE_THROW_UNLESS(row == rows());

  return std::visit(overloaded{
                        [&](const Block3x3SparseMatrix<T>& M) {
                          return MatrixBlock<T>(
                              M.LeftMultiplyByBlockDiagonal(Gs, start, end));
                        },
                        [&](const auto& M) {
                          MatrixX<T> GM(rows(), cols());
                          for (int i = start; i <= end; ++i) {
                            const int rows = Gs[i].rows();
                            const int row_start = row_starts[i - start];
                            GM.middleRows(row_start, rows).noalias() =
                                Gs[i] * M.middleRows(row_start, rows);
                          }
                          return MatrixBlock<T>(std::move(GM));
                        },
                    },
                    data_);
}

template <class T>
void MatrixBlock<T>::MultiplyWithScaledTransposeAndAddTo(
    const VectorX<T>& scale, EigenPtr<MatrixX<T>> y) const {
  DRAKE_THROW_UNLESS(y != nullptr);
  DRAKE_THROW_UNLESS(cols() == scale.size());
  DRAKE_THROW_UNLESS(rows() == y->rows());
  DRAKE_THROW_UNLESS(rows() == y->cols());
  std::visit(overloaded{
                 [&](const Block3x3SparseMatrix<T>& M) {
                   M.MultiplyWithScaledTransposeAndAddTo(scale, y);
                 },
                 [&](const auto& M) {
                   *y += M * scale.asDiagonal() * M.transpose();
                 },
             },
             data_);
}

template <class T>
MatrixBlock<T> MatrixBlock<T>::operator+(const MatrixBlock<T>& other) const {
  DRAKE_THROW_UNLESS(this->rows() == other.rows());
  DRAKE_THROW_UNLESS(this->cols() == other.cols());
  if (this->is_dense() || other.is_dense()) {
    return MatrixBlock<T>(std::get<MatrixX<T>>(this->data_) +
                          std::get<MatrixX<T>>(other.data_));
  }

  int nonzero_entries = 0;
  auto accumulate_nonzero_entries = overloaded{
      [&](const Block3x3SparseMatrix<T>& entry) {
        nonzero_entries += entry.num_blocks() * 9;
      },
      [&](const Eigen::SparseMatrix<T, Eigen::RowMajor>& entry) {
        nonzero_entries += entry.nonZeros();
      },
      [&](const MatrixX<T>&) {
        DRAKE_UNREACHABLE();
      },
  };
  std::visit(accumulate_nonzero_entries, this->data_);
  std::visit(accumulate_nonzero_entries, other.data_);

  std::vector<Eigen::Triplet<T>> triplets;
  triplets.reserve(nonzero_entries);
  auto extract_triplets = overloaded{
      [&](const Block3x3SparseMatrix<T>& entry) {
        for (const auto& row_data : entry.get_triplets()) {
          for (const auto& t : row_data) {
            const int block_row = std::get<0>(t);
            const int block_col = std::get<1>(t);
            const Matrix3<T>& m = std::get<2>(t);
            for (int j = 0; j < 3; ++j) {
              for (int i = 0; i < 3; ++i) {
                triplets.emplace_back(block_row * 3 + i, block_col * 3 + j,
                                      m(i, j));
              }
            }
          }
        }
      },
      [&](const Eigen::SparseMatrix<T, Eigen::RowMajor>& entry) {
        for (int i = 0; i < entry.outerSize(); ++i) {
          for (typename Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator
                   it(entry, i);
               it; ++it) {
            triplets.emplace_back(it.row(), it.col(), it.value());
          }
        }
      },
      [&](const MatrixX<T>&) {
        DRAKE_UNREACHABLE();
      },
  };
  std::visit(extract_triplets, this->data_);
  std::visit(extract_triplets, other.data_);
  Eigen::SparseMatrix<T, Eigen::RowMajor> result(rows(), cols());
  result.setFromTriplets(triplets.begin(), triplets.end());
  return MatrixBlock<T>(std::move(result));
}

template <class T>
MatrixX<T> MatrixBlock<T>::MakeDenseMatrix() const {
  return std::visit(overloaded{
                        [&](const MatrixX<T>& M) {
                          return M;
                        },
                        [&](const Block3x3SparseMatrix<T>& M) {
                          return M.MakeDenseMatrix();
                        },
                        [&](const Eigen::SparseMatrix<T, Eigen::RowMajor>& M) {
                          return M.toDense();
                        },
                    },
                    data_);
}

template <class T>
bool MatrixBlock<T>::operator==(const MatrixBlock<T>& other) const {
  return std::visit(
      [](const auto& a, const auto& b) -> bool {
        if constexpr (std::is_same_v<T, symbolic::Expression>) {
          throw std::logic_error(
              "Comparing MatrixBlock<T> with operator== for"
              " T=symbolic::Expression is invalid.");
        } else {
          using AType = std::decay_t<decltype(a)>;
          using BType = std::decay_t<decltype(b)>;
          if constexpr (std::is_same_v<AType, BType>) {
            if constexpr (std::is_same_v<
                              AType, Eigen::SparseMatrix<T, Eigen::RowMajor>>) {
              return (a - b).norm() == 0.0;
            } else {
              return a == b;
            }
          } else {
            return false;  // mismatched types
          }
        }
      },
      this->data_, other.data_);
}

template <typename T>
MatrixBlock<T> StackMatrixBlocks(const std::vector<MatrixBlock<T>>& blocks) {
  if (blocks.empty()) {
    return {};
  }

  const bool is_dense = blocks[0].is_dense();
  bool is_3x3sparse = true;
  const int cols = blocks[0].cols();
  int rows = 0;
  for (const auto& b : blocks) {
    DRAKE_THROW_UNLESS(is_dense == b.is_dense());
    if (!std::holds_alternative<Block3x3SparseMatrix<T>>(b.data_))
      is_3x3sparse = false;
    DRAKE_THROW_UNLESS(cols == b.cols());
    rows += b.rows();
  }

  if (is_dense) {
    MatrixX<T> result(rows, cols);
    int row_offset = 0;
    for (const auto& b : blocks) {
      result.middleRows(row_offset, b.rows()) = std::get<MatrixX<T>>(b.data_);
      row_offset += b.rows();
    }
    return MatrixBlock<T>(std::move(result));
  }

  if (is_3x3sparse) {
    DRAKE_THROW_UNLESS(rows % 3 == 0);
    DRAKE_THROW_UNLESS(cols % 3 == 0);
    const int block_rows = rows / 3;
    const int block_cols = cols / 3;
    int block_row_offset = 0;
    Block3x3SparseMatrix<T> result(block_rows, block_cols);
    using Triplet = typename Block3x3SparseMatrix<T>::Triplet;
    std::vector<Triplet> result_triplets;
    int nonzero_entries = 0;
    for (const auto& b : blocks) {
      const Block3x3SparseMatrix<T>& entry =
          std::get<Block3x3SparseMatrix<T>>(b.data_);
      nonzero_entries += entry.num_blocks();
    }
    result_triplets.reserve(nonzero_entries);

    for (const auto& b : blocks) {
      const Block3x3SparseMatrix<T>& entry =
          std::get<Block3x3SparseMatrix<T>>(b.data_);
      const std::vector<std::vector<Triplet>>& b_triplets =
          entry.get_triplets();
      for (const auto& row_data : b_triplets) {
        for (const Triplet& t : row_data) {
          const int block_row = std::get<0>(t) + block_row_offset;
          const int block_col = std::get<1>(t);
          const Matrix3<T>& m = std::get<2>(t);
          result_triplets.emplace_back(block_row, block_col, m);
        }
      }
      block_row_offset += entry.block_rows();
    }
    result.SetFromTriplets(result_triplets);
    return MatrixBlock<T>(std::move(result));
  }

  int row_offset = 0;
  Eigen::SparseMatrix<T, Eigen::RowMajor> result(rows, cols);
  std::vector<Eigen::Triplet<T>> result_triplets;
  int nonzero_entries = 0;
  for (const auto& b : blocks) {
    std::visit(overloaded{
                   [&](const Block3x3SparseMatrix<T>& entry) {
                     nonzero_entries += entry.num_blocks() * 9;
                   },
                   [&](const Eigen::SparseMatrix<T, Eigen::RowMajor>& entry) {
                     nonzero_entries += entry.nonZeros();
                   },
                   [&](const MatrixX<T>&) {
                     DRAKE_UNREACHABLE();
                   },
               },
               b.data_);
  }
  result_triplets.reserve(nonzero_entries);

  for (const auto& b : blocks) {
    std::visit(
        overloaded{
            [&](const Block3x3SparseMatrix<T>& entry) {
              for (const auto& row_data : entry.get_triplets()) {
                for (int i = 0; i < 3; ++i) {
                  for (const typename Block3x3SparseMatrix<T>::Triplet& t :
                       row_data) {
                    for (int j = 0; j < 3; ++j) {
                      const int block_row = std::get<0>(t);
                      const int block_col = std::get<1>(t);
                      const Matrix3<T>& m = std::get<2>(t);
                      result_triplets.emplace_back(
                          row_offset + block_row * 3 + i, block_col * 3 + j,
                          m(i, j));
                    }
                  }
                }
              }
            },
            [&](const Eigen::SparseMatrix<T, Eigen::RowMajor>& entry) {
              for (int i = 0; i < entry.outerSize(); ++i) {
                for (typename Eigen::SparseMatrix<
                         T, Eigen::RowMajor>::InnerIterator it(entry, i);
                     it; ++it) {
                  result_triplets.emplace_back(row_offset + it.row(), it.col(),
                                               it.value());
                }
              }
            },
            [&](const MatrixX<T>&) {
              DRAKE_UNREACHABLE();
            },
        },
        b.data_);
    row_offset += b.rows();
  }
  result.setFromTriplets(result_triplets.begin(), result_triplets.end());
  return MatrixBlock<T>(std::move(result));
}

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    (&StackMatrixBlocks<T>));

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::contact_solvers::internal::MatrixBlock);
