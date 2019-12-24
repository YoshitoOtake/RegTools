#ifndef __FileMHD_h
#define __FileMHD_h

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>


/*!
  \class FileMHD
  \author Ali Uneri
  \date 2014-02-08
  \details See http://www.itk.org/Wiki/ITK/MetaIO/Documentation.

  \todo(auneri1) Make these methods feature complete with PythonTools and MatlabTools.
 */
class FileMHD {

  struct Header {
    std::string Comment;                   // MET_STRING
    std::string ObjectType;                // MET_STRING (Image)
    std::string ObjectSubType;             // MET_STRING
    std::string TransformType;             // MET_STRING (Rigid)
    int NDims;                             // MET_INT
    std::string Name;                      // MET_STRING
    int ID;                                // MET_INT
    int ParentID;                          // MET_INT
    bool CompressedData;                   // MET_STRING (boolean)
    int CompressedDataSize;                // MET_INT
    bool BinaryData;                       // MET_STRING (boolean)
    bool BinaryDataByteOrderMSB;           // MET_STRING (boolean)
    bool ElementByteOrderMSB;              // MET_STRING (boolean)
    std::vector<double> Color;             // MET_FLOAT_ARRAY[4]
    std::vector<double> Position;          // MET_FLOAT_ARRAY[NDims]
    std::vector<double> Offset;            // == Position
    std::vector<double> Origin;            // == Position
    std::vector<double> Orientation;       // MET_FLOAT_MATRIX[NDims][NDims]
    std::vector<double> Rotation;          // == Orientation
    std::vector<double> TransformMatrix;   // == Orientation
    std::vector<double> CenterOfRotation;  // MET_FLOAT_ARRAY[NDims]
    std::string AnatomicalOrientation;     // MET_STRING (RAS)
    std::vector<double> ElementSpacing;    // MET_FLOAT_ARRAY[NDims]
    std::vector<int> DimSize;              // MET_INT_ARRAY[NDims]
    int HeaderSize;                        // MET_INT
    std::string Modality;                  // MET_STRING (MET_MOD_CT)
    std::vector<int> SequenceID;           // MET_INT_ARRAY[4]
    double ElementMin;                     // MET_FLOAT
    double ElementMax;                     // MET_FLOAT
    int ElementNumberOfChannels;           // MET_INT
    std::vector<double> ElementSize;       // MET_FLOAT_ARRAY[NDims]
    std::string ElementType;               // MET_STRING (MET_UINT)
    std::string ElementDataFile;           // MET_STRING
  };

 public:
  FileMHD(const std::string &filename) {
    this->filename = filename;
    header.CompressedData = false;
    header.HeaderSize = 0;
  }

  bool Read(std::vector<int> &dimensions, std::vector<float> &values, std::vector<double> &spacing);
  bool Write(const std::vector<int> &dimensions, const std::vector<float> &values, const std::vector<double> &spacing);

  std::string filename;
  Header header;

 private:
  void Strip(std::string &s);

  template <class FromType, class ToType>
  struct StaticCaster {
    ToType operator()(FromType p) { return static_cast<ToType>(p); }
  };

  template <class FromType, class ToType>
  void ReadCast(std::ifstream &file, std::vector<float> &values, const unsigned int size) {
    values.clear();
    values.reserve(size);
    std::vector<FromType> data(size);
    file.read(reinterpret_cast<char*>(&(data.front())), size * sizeof(FromType));
    std::transform(data.begin(), data.end(), std::back_inserter(values), StaticCaster<FromType,ToType>());
  }
};


#endif  // __FileMHD_h
