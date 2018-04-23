#include<iostream>
#include<set>
#include<vector>
#include<algorithm>
#include<fstream>
#include<math.h>

using namespace std;

#define RobotNum 20
#define MapSize 10
#define Distance 4.1
#define Beacon1ID 0
#define Beacon2ID 1
#define Beacon3ID 16


class Robot{
public:
	Robot(){

	}
	Robot(int x, int y){
		this->locationx = x;
		this->locationy = y;
	}
	void setPos(int x, int y){
		this->locationx = x;
		this->locationy = y;
	}
	int id;
	float locationx;
	float locationy;
	int state = 0;
	vector<pair<int, float> > myNeighbor;
	vector<float > everyPoint_distance;
	vector<int > everyPoint_hop;
};

//产生MapSize大小的地图中产生RobotNum个不重复的随机点
void rand2array(float *PointX, float *PointY) {
	//	int PointX[MapSize],PointY[MapSize];
	set<pair<float, float> > s;
	static int m = 10000;
	srand(m--);
	while (s.size() < RobotNum) {
		float x = (rand() % (MapSize*100))/100.0;
		float y = (rand() % (MapSize * 100)) / 100.0;
		s.insert(pair<float, float>(x, y));
	}
	int i = 0;
	for (set<pair<float, float> >::iterator iter = s.begin(); iter != s.end(); iter++, i++) {
		PointX[i] = iter->first;
		PointY[i] = iter->second;
	}
}

bool cmp_by_value(const pair<int, float>  &lhs, const pair<int, float > &rhs) {
	return lhs.second < rhs.second;
}


void createMap(Robot *probot) {
	float PointX[RobotNum], PointY[RobotNum];
	rand2array(PointX, PointY);
	for (int i = 0; i < RobotNum; i++) {
		probot[i].id = i;
		probot[i].locationx = PointX[i];
		probot[i].locationy = PointY[i];
	}

	for (int i = 0; i < RobotNum; i++) {
		cout << "Robot" << i << " Position is (" << probot[i].locationx << "," << probot[i].locationy << ")" << endl;
	}
	//求各点之间相互距离小于Distance的点存入myNeighborId中
	for (int i = 0; i < RobotNum; i++) {
		for (int j = i + 1; j < RobotNum; j++) {
			float tempDistance = sqrt((probot[i].locationx - probot[j].locationx)*(probot[i].locationx - probot[j].locationx) + (probot[i].locationy - probot[j].locationy)*(probot[i].locationy - probot[j].locationy));
			if (tempDistance < Distance) {
				probot[i].myNeighbor.push_back(pair<int, float>(j, tempDistance));
				probot[j].myNeighbor.push_back(pair<int, float>(i, tempDistance));
			}
		}
	}

	for (int i = 0; i < RobotNum; i++){
		sort(probot[i].myNeighbor.begin(), probot[i].myNeighbor.end(), cmp_by_value);
	}
	for (int i = 0; i < RobotNum; i++) {
		cout << "Robot" << i << " have neighbor ";
		for (int j = 0; j < probot[i].myNeighbor.size(); j++){
			cout << probot[i].myNeighbor[j].first << " ";
//			cout << probot[i].myNeighbor[j].second << " ";
		}
			
		cout << endl;
	}

	//初始化所有点之间的距离，跳数为0。
	for (int i = 0; i < RobotNum; i++){
		for (int j = 0; j < RobotNum; j++){
			probot[i].everyPoint_distance.push_back(999);
			probot[i].everyPoint_hop.push_back(999);
			if (i == j){
				probot[i].everyPoint_hop[j] = 0;
				probot[i].everyPoint_distance[j] = 0;
			}
		}
	}
	for (int i = 0; i < RobotNum; i++){
		for (int j = 0; j < probot[i].myNeighbor.size(); j++){
			int nei_id = probot[i].myNeighbor[j].first;
			float nei_distance = probot[i].myNeighbor[j].second;
			
			probot[i].everyPoint_distance[nei_id] = nei_distance;
			probot[i].everyPoint_hop[nei_id] = 1;
		}
	}
	for (int i = 0; i < RobotNum; i++){
		for (int j = 0; j < RobotNum; j++){
			cout << probot[i].everyPoint_hop[j] << "\t";
		}
		cout << endl;
	}
}

int main(){
	
	ofstream point, hop, distance;
	point.open("point.data", ios::out);
	hop.open("hop.data", ios::out);
	distance.open("distance.data", ios::out);
	int count = 1000;
	;
	while (count--){
		Robot *probot = new Robot[RobotNum];

		probot[Beacon1ID].state = 3;
		probot[Beacon2ID].state = 3;
		probot[Beacon3ID].state = 3;
		createMap(probot);

		for (int i = 0; i < RobotNum; i++){
			for (int j = 0; j < RobotNum; j++){
				for (int k = 0; k < RobotNum; k++){
					if (probot[j].everyPoint_hop[k]>probot[j].everyPoint_hop[i] + probot[i].everyPoint_hop[k])
						probot[j].everyPoint_hop[k] = probot[j].everyPoint_hop[i] + probot[i].everyPoint_hop[k];

				}
			}
		}
		for (int i = 0; i < RobotNum; i++){
			for (int j = 0; j < RobotNum; j++){
				for (int k = 0; k < RobotNum; k++){
					if (probot[j].everyPoint_distance[k]>probot[j].everyPoint_distance[i] + probot[i].everyPoint_distance[k])
						probot[j].everyPoint_distance[k] = probot[j].everyPoint_distance[i] + probot[i].everyPoint_distance[k];

				}
			}
		}

		cout << "another train --- hop\n";

		for (int i = 0; i < RobotNum; i++){
			for (int j = 0; j < RobotNum; j++){
				printf("%d\t", probot[i].everyPoint_hop[j]);
				if (probot[i].everyPoint_hop[j] >990)
					hop << -1 << " ";
				else
					hop << probot[i].everyPoint_hop[j] << " ";
			}
			hop << endl;
			printf("\n");
		}

		cout << "another train --- distance\n";

		for (int i = 0; i < RobotNum; i++){
			for (int j = 0; j < RobotNum; j++){
				printf("%.2f\t", probot[i].everyPoint_distance[j]);
				if (probot[i].everyPoint_distance[j] >990)
					distance << -1 << " ";
				else
					distance << probot[i].everyPoint_distance[j] << " ";
			}
			distance << endl;
			printf("\n");
		}
		cout << "another train --- point\n";
		for (int i = 0; i < RobotNum; i++){
			point << probot[i].locationx << " " << probot[i].locationy << endl;
		}
		point << endl << endl;
		hop << endl << endl;
		distance << endl << endl;
		delete[] probot;
	}
	return 0;
}

/*
while (1){
for (int i = 0; i < RobotNum; i++){
for (int j = 0; j < probot[i].myNeighbor.size(); j++){
int nei_id = probot[i].myNeighbor[j].first;

//把邻居所知的距离，跳数都告诉本节点。当本结点到新节点距离为0，直接更新。当距离不为0，
//比较本结点到这个节点的距离和邻居节点到这个点距离+邻居节点到本节点距离,若新的小，更新。
float nei_distance = probot[i].everyPoint_distance[nei_id];

for (int k = 0; k < probot[nei_id].everyPoint_distance.size(); k++){
if (i == k)
break;
float nei_nei_distance = probot[nei_id].everyPoint_distance[k];//第k个节点到第nei_id个点的距离
int nei_nei_hop = probot[nei_id].everyPoint_hop[k];//第k个节点到第nei_id个点的跳数
if (probot[i].everyPoint_distance[k] == 0 && nei_nei_distance != 0){
probot[i].everyPoint_distance[k] = nei_distance + nei_nei_distance;
}
else{
if (probot[i].everyPoint_distance[k] > nei_distance + nei_nei_distance){
probot[i].everyPoint_distance[k] = nei_distance + nei_nei_distance;
}
}
if (probot[i].everyPoint_hop[k] == 0 && nei_nei_hop != 0){
probot[i].everyPoint_hop[k] = nei_nei_hop + 1;
}
else{
if (probot[i].everyPoint_hop[k] > nei_nei_hop + 1 && nei_nei_hop != 0){
probot[i].everyPoint_hop[k] = nei_nei_hop + 1;
}
}
}
}
}
*/
