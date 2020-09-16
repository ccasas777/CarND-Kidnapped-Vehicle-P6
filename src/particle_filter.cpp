/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>


#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  
  num_particles = 100;  // TODO: Set the number of particles
  std::default_random_engine gen;
  // This line creates a normal (Gaussian) distribution for x,y,theta
  normal_distribution<double> dist_x(x, std[0]);  
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {
      Particle current_particle;
	  current_particle.id = i;
	  current_particle.x = dist_x(gen);
	  current_particle.y = dist_y(gen);
	  current_particle.theta = dist_theta(gen);
	  current_particle.weight = 1.0;
	  
	  particles.push_back(current_particle);
	       
    
  }
  is_initialized = true;
  
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  
  std::default_random_engine gen;
  normal_distribution<double> err_x(0, std_pos[0]);  
  normal_distribution<double> err_y(0, std_pos[1]);
  normal_distribution<double> err_theta(0, std_pos[2]);
  
  
  for(int i=0; i<num_particles;i++){
    
    if (fabs(yaw_rate)<0.0001){
      particles[i].x += velocity * delta_t * cos( particles[i].theta );
      particles[i].y += velocity * delta_t * sin( particles[i].theta );
       // yaw continue to be the same.
    }else{
      particles[i].x += velocity/yaw_rate*(sin(particles[i].theta + yaw_rate* delta_t)-sin(particles[i].theta));
      particles[i].y += velocity/yaw_rate*(-cos(particles[i].theta + yaw_rate* delta_t)+cos(particles[i].theta)); 
      particles[i].theta += yaw_rate* delta_t; 
      if (particles[i].theta > 2*M_PI){particles[i].theta-=2*M_PI;}
      if (particles[i].theta < -2*M_PI){particles[i].theta+=2*M_PI;}
      
    }
    
    
    // Adding noise.
    particles[i].x += err_x(gen);
    particles[i].y += err_y(gen);
    particles[i].theta += err_theta(gen);
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  
    unsigned int ob_size = observations.size();
    unsigned int pre_size = predicted.size();

    for (unsigned int i = 0; i < ob_size; i++) { // For each observation

      // Initialize min distance as a really big number.
      double minDistance = 50;

      // Initialize the found map in something not possible.
      int mapId = -1;

      for (unsigned j = 0; j < pre_size; j++ ) { // For each predition.

        double xDistance = observations[i].x - predicted[j].x;
        double yDistance = observations[i].y - predicted[j].y;

        double distance = xDistance * xDistance + yDistance * yDistance;

        // If the "distance" is less than min, stored the id and update min.
        // in the result, getting the nearest neighbor of landmark id for this measurement.
        if ( distance < minDistance ) {
          minDistance = distance;
          mapId = predicted[j].id;
        }
      }
      // Update the observation identifier.
      observations[i].id = mapId;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

    for (int i = 0; i < num_particles; i++) {

      double x = particles[i].x;
      double y = particles[i].y;
      double theta = particles[i].theta;
      
      
      // Find landmarks in particle's range.
      double sensor_range_2 = sensor_range * sensor_range;
      vector<LandmarkObs> inRange_Landmarks;
      for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
        float landmark_X = map_landmarks.landmark_list[j].x_f;
        float landmark_Y = map_landmarks.landmark_list[j].y_f;
        int id = map_landmarks.landmark_list[j].id_i;
        double dX = x - landmark_X;
        double dY = y - landmark_Y;
        if ( dX*dX + dY*dY <= sensor_range_2 ) {
          inRange_Landmarks.push_back(LandmarkObs{ id, landmark_X, landmark_Y });
        }
      }
      
      
     // Transform observation coordinates.
     // "observations" is measured according to the car coordinate, which is need to transform to particle coordinate. 
    vector<LandmarkObs> trans_observations;
    for(unsigned int j = 0; j < observations.size(); j++) {
      double xx = cos(theta)*observations[j].x - sin(theta)*observations[j].y + x;
      double yy = sin(theta)*observations[j].x + cos(theta)*observations[j].y + y;
      trans_observations.push_back(LandmarkObs{ observations[j].id, xx, yy });
    }

    // Observation association to landmark.
    // find the corresponding id for the trans_observation.  
    dataAssociation(inRange_Landmarks, trans_observations);

      
      
    // Reseting weight.
    particles[i].weight = 1.0;
    // Calculate weights.
    for(unsigned int j = 0; j < trans_observations.size(); j++) {
      double obs_X = trans_observations[j].x;
      double obs_Y = trans_observations[j].y;

      int obs_Id = trans_observations[j].id;

      double landmark_X, landmark_Y;
      unsigned int k = 0;
      unsigned int nLandmarks = inRange_Landmarks.size();
      
      bool found = false;   
      while( !found && k < nLandmarks ) {
        if ( inRange_Landmarks[k].id == obs_Id) {
          found = true;
          landmark_X = inRange_Landmarks[k].x;
          landmark_Y = inRange_Landmarks[k].y;
        }
        k++;
      }

      // Calculating weight.
      double dX = obs_X - landmark_X;
      double dY = obs_Y - landmark_Y;

      double weight = ( 1/(2*M_PI*std_landmark[0]*std_landmark[1])) * exp( -( dX*dX/(2*std_landmark[0]*std_landmark[0]) + (dY*dY/(2*std_landmark[1]*std_landmark[1])) ) );
      if (weight == 0) {
        particles[i].weight *= 0.0001;
      } else {
        particles[i].weight *= weight;
      }
    }
  }
  
  
  
  
}

void ParticleFilter::resample() {

  
    // Get weights and max weight.
  vector<double> weights;
  double maxWeight = -1;
  for(int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
    if ( particles[i].weight > maxWeight ) {
      maxWeight = particles[i].weight;
    }
  }
  
  std::default_random_engine gen;
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  std::uniform_int_distribution<int> index(0,num_particles-1);
  int ind = index(gen);
  
  vector<Particle> new_particles ;
  double beta = 0;
  for (int i = 0; i < num_particles; ++i) {
    
    
     beta += maxWeight*2*unif(gen);
    
    while(beta > weights[ind]){
      beta -= weights[ind];
      ind = (ind+1) % num_particles;      
    }
        
    new_particles.push_back(particles[ind]);
       
  }
particles = new_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();
  
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}