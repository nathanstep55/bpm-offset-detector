#pragma once

// https://web.archive.org/web/20190201175919/http://www.vilipetek.com/2013/10/17/polynomial-fitting-in-c-not-using-boost/
// Originally sourced from Vili Petek, modified by Bram van de Wetering

#include <vector>
#include <algorithm>
#include <math.h>

namespace mathalgo {

using namespace std;

typedef unsigned int uint;

template <typename T>
struct matrix
{
	matrix(uint nRows, uint nCols) :
		rows(nRows),
		cols(nCols),
		data(nRows * nCols, 0)
	{
	}
	static matrix identity(uint nSize)
	{
		matrix oResult(nSize, nSize);

		int nCount = 0;
		std::generate(oResult.data.begin(), oResult.data.end(),
			[&nCount, nSize]() { return !(nCount++ % (nSize + 1)); });

		return oResult;
	}
	inline T& operator()(uint nRow, uint nCol)
	{
		return data[nCol + cols*nRow];
	}
	inline matrix operator*(matrix& other)
	{
		matrix oResult(rows, other.cols);
		for(uint r = 0; r < rows; ++r)
		{
			for(uint ocol = 0; ocol < other.cols; ++ocol)
			{
				for(uint c = 0; c < cols; ++c)
				{
					oResult(r, ocol) += (*this)(r, c) * other(c, ocol);
				}
			}
		}
		return oResult;
	}
	inline matrix transpose()
	{
		matrix oResult(cols, rows);
		for(uint r = 0; r < rows; ++r)
		{
			for(uint c = 0; c < cols; ++c)
			{
				oResult(c, r) += (*this)(r, c);
			}
		}
		return oResult;
	}
	std::vector<T> data;
	uint rows;
	uint cols;
};

template <typename T>
struct Givens
{
public:
	Givens() : m_oJ(2, 2), m_oQ(1, 1), m_oR(1, 1)
	{
	}

	/*
	Calculate the inverse of a matrix using the QR decomposition.

	param:
	A	matrix to inverse
	*/
	const matrix<T> Inverse(matrix<T>& oMatrix)
	{
		matrix<T> oIdentity = matrix<T>::identity(oMatrix.rows());
		Decompose(oMatrix);
		return Solve(oIdentity);
	}

	/*
	Performs QR factorization using Givens rotations.
	*/
	void Decompose(matrix<T>& oMatrix)
	{
		int nRows = oMatrix.rows;
		int nCols = oMatrix.cols;


		if(nRows == nCols)
		{
			nCols--;
		}
		else if(nRows < nCols)
		{
			nCols = nRows - 1;
		}

		m_oQ = matrix<T>::identity(nRows);
		m_oR = oMatrix;

		for(int j = 0; j < nCols; j++)
		{
			for(int i = j + 1; i < nRows; i++)
			{
				GivensRotation(m_oR(j, j), m_oR(i, j));
				PreMultiplyGivens(m_oR, j, i);
				PreMultiplyGivens(m_oQ, j, i);
			}
		}

		m_oQ = m_oQ.transpose();
	}

	/*
	Find the solution for a matrix.
	http://en.wikipedia.org/wiki/QR_decomposition#Using_for_solution_to_linear_inverse_problems
	*/
	matrix<T> Solve(matrix<T>& oMatrix)
	{
		matrix<T> oQtM(m_oQ.transpose() * oMatrix);
		int nCols = m_oR.cols;
		matrix<T> oS(1, nCols);
		for(int i = nCols - 1; i >= 0; i--)
		{
			oS(0, i) = oQtM(i, 0);
			for(int j = i + 1; j < nCols; j++)
			{
				oS(0, i) -= oS(0, j) * m_oR(i, j);
			}
			oS(0, i) /= m_oR(i, i);
		}

		return oS;
	}

	const matrix<T>& GetQ()
	{
		return m_oQ;
	}

	const matrix<T>& GetR()
	{
		return m_oR;
	}

private:
	/*
	Givens rotation is a rotation in the plane spanned by two coordinates axes.
	http://en.wikipedia.org/wiki/Givens_rotation
	*/
	void GivensRotation(T a, T b)
	{
		T t, s, c;
		if(b == 0)
		{
			c = (a >= 0) ? T(1) : T(-1);
			s = 0;
		}
		else if(a == 0)
		{
			c = 0;
			s = (b >= 0) ? T(-1) : T(1);
		}
		else if(abs(b) > abs(a))
		{
			t = a / b;
			s = -1 / sqrt(1 + t*t);
			c = -s*t;
		}
		else
		{
			t = b / a;
			c = 1 / sqrt(1 + t*t);
			s = -c*t;
		}
		m_oJ(0, 0) = c; m_oJ(0, 1) = -s;
		m_oJ(1, 0) = s; m_oJ(1, 1) = c;
	}

	/*
	Get the premultiplication of a given matrix
	by the Givens rotation.
	*/
	void PreMultiplyGivens(matrix<T>& oMatrix, int i, int j)
	{
		int nRowSize = oMatrix.cols;

		for(int nRow = 0; nRow < nRowSize; nRow++)
		{
			double nTemp = oMatrix(i, nRow) * m_oJ(0, 0) + oMatrix(j, nRow) * m_oJ(0, 1);
			oMatrix(j, nRow) = oMatrix(i, nRow) * m_oJ(1, 0) + oMatrix(j, nRow) * m_oJ(1, 1);
			oMatrix(i, nRow) = T(nTemp);
		}
	}

private:
	matrix<T> m_oQ, m_oR, m_oJ;
};

/*
	Finds the coefficients of a polynomial p(x) of degree n that fits the data, 
	p(x(i)) to y(i), in a least squares sense. The result p is a row vector of 
	length n+1 containing the polynomial coefficients in incremental powers.

	param:
		oX				x axis values
		oY				y axis values
		nDegree			polynomial degree including the constant

	return:
		coefficients of a polynomial starting at the constant coefficient and
		ending with the coefficient of power to nDegree. C++0x-compatible 
		compilers make returning locally created vectors very efficient.

*/
template<typename T>
std::vector<T> polyfit(const T* oX, const T* oY, size_t nCount, int nDegree)
{
	// more intuative this way
	nDegree++;

	matrix<T> oXMatrix( nCount, nDegree );
	matrix<T> oYMatrix( nCount, 1 );

	// copy y matrix
	for ( size_t i = 0; i < nCount; i++ )
	{
		oYMatrix(i, 0) = oY[i];
	}

	// create the X matrix
	for ( size_t nRow = 0; nRow < nCount; nRow++ )
	{
		T nVal = 1.0f;
		for ( int nCol = 0; nCol < nDegree; nCol++ )
		{
			oXMatrix(nRow, nCol) = nVal;
			nVal *= oX[nRow];
		}
	}

	// transpose X matrix
	matrix<T> oXtMatrix( oXMatrix.transpose() );
	// multiply transposed X matrix with X matrix
	matrix<T> oXtXMatrix( oXtMatrix * oXMatrix );
	// multiply transposed X matrix with Y matrix
	matrix<T> oXtYMatrix( oXtMatrix * oYMatrix );

	Givens<T> oGivens;
	oGivens.Decompose( oXtXMatrix );
	matrix<T> oCoeff = oGivens.Solve( oXtYMatrix );
	// copy the result to coeff
	return oCoeff.data();
}

// Specialized version for BPM testing, writes degree + 1 coefficients to outCoefs.
template <typename T>
void polyfit(int degree, T* outCoefs, const T* inValues, size_t numNonZeroValues, int offsetX)
{
	// more intuitive this way
	++degree;

	matrix<T> oXMatrix(numNonZeroValues, degree);
	matrix<T> oYMatrix(numNonZeroValues, 1);

	// copy y matrix
	for(size_t nRow = 0, i = 0; nRow < numNonZeroValues; ++nRow, ++i)
	{	
		while(inValues[i] == 0) ++i;
		//printf("%f\n", inValues[i]);
		oYMatrix(nRow, 0) = inValues[i];
	}

	// create the X matrix
	for(size_t nRow = 0, i = 0; nRow < numNonZeroValues; ++nRow, ++i)
	{
		while(inValues[i] == 0) ++i;
		T nVal = 1.0f, x = T(offsetX + i);
		//printf("%f\n", x);
		for(int nCol = 0; nCol < degree; nCol++)
		{
			oXMatrix(nRow, nCol) = nVal;
			nVal *= x;
		}
	}

	// transpose X matrix
	matrix<T> oXtMatrix(oXMatrix.transpose());
	// multiply transposed X matrix with X matrix
	matrix<T> oXtXMatrix(oXtMatrix * oXMatrix);
	// multiply transposed X matrix with Y matrix
	matrix<T> oXtYMatrix(oXtMatrix * oYMatrix);

	Givens<T> oGivens;
	oGivens.Decompose(oXtXMatrix);
	matrix<T> oCoeff = oGivens.Solve(oXtYMatrix);

	// copy the result to coeff
	for(int i = 0; i < degree; ++i)
		outCoefs[i] = oCoeff.data[i];
}

/*
	Calculates the value of a polynomial of degree n evaluated at x. The input 
	argument pCoeff is a vector of length n+1 whose elements are the coefficients 
	in incremental powers of the polynomial to be evaluated.

	param:
		oCoeff			polynomial coefficients generated by polyfit() function
		oX				x axis values

	return:
		Fitted Y values. C++0x-compatible compilers make returning locally 
		created vectors very efficient.
*/
template<typename T>
std::vector<T> polyval(const std::vector<T>& oCoeff, const T* oX, size_t nCount)
{
	size_t nDegree = oCoeff.size();
	std::vector<T>	oY( nCount );

	for ( size_t i = 0; i < nCount; i++ )
	{
		T nY = 0;
		T nXT = 1;
		T nX = oX[i];
		for ( size_t j = 0; j < nDegree; j++ )
		{
			// multiply current x by a coefficient
			nY += oCoeff[j] * nXT;
			// power up the X
			nXT *= nX;
		}
		oY[i] = nY;
	}

	return oY;
}

}; // namespace mathalgo