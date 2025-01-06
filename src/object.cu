#include <object.cuh>
#include <set>
#include <vector3.cuh>
#include <utility.cuh>

#include <stdexcept>
#include <sstream>
#include <string>
#include <fstream>
#include <vector>
#include <limits>

namespace render
{
const Vector3f EDGE_BASE_COLOR(0.25f, 0.25f, 0.25f);
const Vector3f CORNER_BASE_COLOR(0.25f, 0.25f, 0.25f);
void Object::CreateObjectFromFile(const std::string& path, std::vector<Polygon>& polygons)
{
    std::ifstream file(path);
    if(!file) {
        throw std::runtime_error{"File " + path + " not found"};
    }
    std::string line;
    std::vector<Polygon> tempPolygons;
    std::vector<Vector3d> vertices;
    std::vector<std::set<int>> polygonsByVertex;

    int polygonId{0};

    while(std::getline(file, line))
    {
        std::istringstream ss(line);
        std::string type;
        ss >> type;
        if(type == "v")
        {
            Vector3d vertex;
            ss >> vertex;
            vertex *= radius_m;
            vertices.push_back(std::move(vertex));
            polygonsByVertex.emplace_back();
        }
        else if(type == "f") 
        {
            std::string v1, v2, v3, v4;
            ss >> v1 >> v2 >> v3 >> v4;
            if(!v4.empty())
            {
                auto vertexId1 = SplitString(v1, '/');
                auto vertexId2 = SplitString(v2, '/');
                auto vertexId3 = SplitString(v3, '/');
                auto vertexId4 = SplitString(v4, '/');

                --vertexId1;
                --vertexId2;
                --vertexId3;
                --vertexId4;

                tempPolygons.emplace_back(
                    Triangle(vertices[vertexId3], vertices[vertexId4], vertices[vertexId1]), 
                    Vector3f(0.0, 0.0, 0.0));
                tempPolygons.emplace_back(
                    Triangle(vertices[vertexId1], vertices[vertexId2], vertices[vertexId3]), 
                    Vector3f(0.0, 0.0, 0.0));

                polygonsByVertex[vertexId3].insert(polygonId);
                polygonsByVertex[vertexId4].insert(polygonId);
                polygonsByVertex[vertexId1].insert(polygonId);

                ++polygonId;

                polygonsByVertex[vertexId1].insert(polygonId);
                polygonsByVertex[vertexId2].insert(polygonId);
                polygonsByVertex[vertexId3].insert(polygonId);

                ++polygonId;
            }
            else
            {
                auto vertexId1 = SplitString(v1, '/');
                auto vertexId2 = SplitString(v2, '/');
                auto vertexId3 = SplitString(v3, '/');

                --vertexId1;
                --vertexId2;
                --vertexId3;

                tempPolygons.emplace_back(
                    Triangle(vertices[vertexId1], vertices[vertexId2], vertices[vertexId3]), 
                    Vector3f(0.0, 0.0, 0.0));

                polygonsByVertex[vertexId1].insert(polygonId);
                polygonsByVertex[vertexId2].insert(polygonId);
                polygonsByVertex[vertexId3].insert(polygonId);

                ++polygonId;
            }
        }
    }

    double edgeSize = std::numeric_limits<double>::max();
    for(size_t i = 0; i < vertices.size(); ++i)
    {
        for(size_t j = i + 1; j < vertices.size(); ++j)
        {
            edgeSize = std::min(edgeSize, (vertices[i] - vertices[j]).length());
        }
    }

    std::set<int> usedPolygonId;
    for(size_t i = 0; i < vertices.size(); ++i)
    {
        for(size_t j = i + 1; j < vertices.size(); ++j)
        {
            if((vertices[i] - vertices[j]).length() > edgeSize + EPS)
            {
                continue;
            }

            std::vector<int> sharedPolygonIds;
            std::vector<Triangle> sharedTriangles;

            for (int polyId : polygonsByVertex[i])
            {
                if (polygonsByVertex[j].count(polyId))
                {
                    sharedPolygonIds.push_back(polyId);
                    sharedTriangles.push_back(tempPolygons[polyId].triangle());
                }
            }

            if (sharedTriangles.size() != 2)
            {
                continue;
            }

            Triangle t1 = sharedTriangles[0];
            Triangle t2 = sharedTriangles[1];

            Vector3d n1 = 0.1 * t1.n();
            Vector3d n2 = 0.1 * t2.n();
            Vector3d nAvg = (n1 + n2) / 2.0;

            t1.shift(n1);
            t2.shift(n2);

            Vector3d vi1 = vertices[i] + n1;
            Vector3d vi2 = vertices[i] + n2;
            Vector3d vj1 = vertices[j] + n1;
            Vector3d vj2 = vertices[j] + n2;
            Vector3d viAvg = (vi1 + vi2) / 2 + center_m;
            Vector3d vjAvg = (vj1 + vj2) / 2 + center_m;

            auto firstEdge = Triangle(vi1, vj2, vi2);
            auto secondEdge = Triangle(vi1, vj1, vj2);
            firstEdge.rotate(nAvg);
            secondEdge.rotate(nAvg);

            double t = Polygon(firstEdge, EDGE_BASE_COLOR).RayIntersection(
                Ray({0.0, 0.0, 0.0}, vertices[i], 0));
            auto cornerI = Triangle(vi1, vi2, t * vertices[i] / vertices[i].length());
            cornerI.rotate(nAvg);

            firstEdge.shift(center_m);
            cornerI.shift(center_m);

            polygons.push_back(
                Polygon(firstEdge, EDGE_BASE_COLOR, lightCount_m, viAvg, vjAvg));
            polygons.push_back(Polygon(cornerI, CORNER_BASE_COLOR));

            t = Polygon(secondEdge, EDGE_BASE_COLOR).RayIntersection(
                Ray({0.0, 0.0, 0.0}, vertices[j], 0));
            auto cornerJ = Triangle(vj1, t * vertices[j] / vertices[j].length(), vj2);
            cornerJ.rotate(nAvg);

            secondEdge.shift(center_m);
            cornerJ.shift(center_m);

            polygons.push_back(
                Polygon(secondEdge, EDGE_BASE_COLOR, lightCount_m, viAvg, vjAvg));
            polygons.push_back(Polygon(cornerJ, CORNER_BASE_COLOR));

            auto firstId = sharedPolygonIds[0];
            auto secondId = sharedPolygonIds[1];
            if(usedPolygonId.find(firstId) == usedPolygonId.end())
            {
                t1.shift(center_m);
                polygons.push_back(
                    Polygon(t1, color_m, reflection_m, transparency_m));
                usedPolygonId.insert(firstId);
            }

            if(usedPolygonId.find(secondId) == usedPolygonId.end())
            {
                t2.shift(center_m);
                polygons.push_back(
                    Polygon(t2, color_m, reflection_m, transparency_m));
                usedPolygonId.insert(secondId);
            }
        }
    }

}
};