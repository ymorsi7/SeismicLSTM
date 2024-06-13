# Application of Long Short-Term Memory (LSTM) Networks-Based Surrogate Modeling for Nonlinear Structural Systems

## Abdoul Aziz Sandotin Coulibaly 
## Enrique Simbort Zeballos 
## Yusuf Morsi 
## Ramin Sarange

## Abstract
Performance-based seismic design (PBSD) of structural systems relies on computationally expensive high-fidelity finite element (FE) models to predict how structures will respond to seismic excitation. For risk-based assessments, FE response simulations must be run thousands of times with different realizations of the sources of uncertainty. Consequently, data-driven machine learning (DDML) surrogate models have gained prominence as fast emulators for predicting seismic structural responses in probabilistic analyses. This paper leverages deep Long Short-Term Memory (LSTM) networks, known for their powerful and flexible framework for time series prediction tasks. The advantages of using LSTM networks include their ability to model continuous-time processes, adapt to varying temporal resolutions, maintain implicit memory of past information, model complex nonlinear dynamics, perform interpolation and extrapolation, handle noisy data robustly, and scale effectively to high-dimensional datasets. The effectiveness of the proposed method is validated through three proof-of-concept studies: one involving a linear elastic 2D 8-degree-of-freedom (DoF) shear building model, a nonlinear single degree of freedom system (NL-SDoF), and a 2D nonlinear 3DoF shear building model. The findings indicate that the proposed LSTM network is a promising, dependable, and computationally efficient technique for predicting nonlinear structural responses.

## Introduction
The earthquake events in Afghanistan, Morocco, and Turkey in 2023 led to 63,701 fatalities. These events underscore the necessity for a more refined approach to predicting the dynamic response behavior of structures subjected to large seismic events. This is particularly critical for complex civil infrastructure such as long-span bridges and concrete dams, where the consequences of failure are deemed unacceptable. The 2021 report card released by the American Society of Civil Engineers (ASCE) assigned a C- grade to the U.S. infrastructure, highlighting the need for structural assessment (SA) of critical civil infrastructure systems. However, evaluating the seismic response of long-span bridges or dams subjected to moderate to strong intensity ground motions poses a considerable challenge due to the lack of experimental data and the high computational cost associated with the validation process of state-of-the-art finite element (FE) models, which often rely on Bayesian inference.

## Related Work
Traditional methods for analyzing structural dynamic responses to natural hazards typically use physics-based models, such as the finite element method (FEM). FEM requires fine mesh discretization and small time steps for accuracy, making it computationally expensive, especially for nonlinear time history analysis and when numerous simulations are needed to account for stochastic uncertainties in external loads. Additionally, model assumptions and parameter uncertainties can lead to significant errors.

Alternative approaches, such as system identification (SI) methods, perform linear/nonlinear mapping of external excitations to structural responses using state-space or black-box models. However, these methods often struggle with nonlinear responses due to assumptions of stationary dynamics and linearity. Artificial neural networks (ANNs), particularly multi-layer perceptrons (MLPs), have been used successfully for modeling structural responses under various loads but have limitations with complex time series data.

Recent advances in deep learning, such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs), offer better capabilities for modeling nonlinear structural responses. Despite CNNs' strength in data classification with grid-like topology, they struggle with large plastic deformations. RNNs are designed for sequential, time-varying data but face issues with long-range dependencies due to gradient vanishing and exploding.

This work reproduces an existing machine learning structural engineering application, addressing the limitations described above by developing a long short-term memory (LSTM) network for nonlinear structural response modeling extended to include strong motion datasets. Based on the studied paper, this work introduces one LSTM scheme for seismic response modeling. The effectiveness of the proposed method is validated by three proof-of-concept studies: 2D linear elastic 8 degree-of-freedom (DoF) shear building model, a nonlinear single degree of freedom system (NL-SDoF), and a 2D nonlinear 3DoF shear building model.

## Methodology

### Structural Analysis and Computational Mechanics
Consider the system of second-order differential equations of motion governing the response of a multi-degree of freedom (MDF) system to earthquake-induced ground motion. The objective is to find a reduced-order model based on the deep LSTM networks approach. The methodology is validated through three study examples involving different structural models.

### Long Short-Term Memory (LSTM) Network

#### The FOM Implementation
The FOM with 8 DOFs is used to investigate the numerical performance of the LSTM Network. The system under study is governed by specific differential equations with initial conditions. The FOM was developed in OpenSees as a 2-D linear elastic model of this shear building model system, using fiber-section force-based Bernoulli-Euler beam-column elements. The modal and material properties of models two and three are presented.

Models Two and Three are governed by similar equations of motion, incorporating material nonlinearity. The LSTM network architecture is designed to handle sequential data by maintaining a memory of previous inputs. This architecture includes multiple LSTM cells arranged in layers to capture complex temporal patterns.

#### The LSTM Cell
The LSTM cell consists of several gates that control the flow of information, making it capable of learning long-term dependencies. An LSTM Cell at a specific time step receives an input vector, the hidden state output of the previous step, and the cell memory state of the previous time step. Mathematical relationships between the gates and states are provided.

#### The LSTM Network Architecture
The full sequence-to-sequence LSTM network (LSTM-f) processes entire sequences of data from input to output, effective for tasks requiring accurate prediction of time-series data.

## Experiments

### Experimental Setup
The experimental setup involves preparing data, scaling both input and output data using the MinMaxScaler, and reshaping and scaling data to fit LSTM network requirements. The model architecture includes two LSTM layers with 100 units each, two dense layers, Adam optimizer, a learning rate of 0.001, and MSE loss function. Training and validation losses are monitored.

### Results for the Linear Elastic 8-Story Shear Building
Results are presented for a linear elastic 8-story shear building, showing total acceleration versus time for different epochs of training.

### Results for the Nonlinear Inelastic Single Degree Of Freedom (SDOF) Structure
Results for a nonlinear inelastic SDOF system show predictions with bandlimited white noise as an input excitation and total acceleration as the output response for different epochs of training.

### Results for the Nonlinear Inelastic Multi Degree Of Freedom (MDOF) Structure
Results for the MDOF system show total acceleration and displacement responses at each floor for different epochs of training.

### A/B Testing
A/B testing compares two distinct LSTM models, a simpler architecture (Model A) and a more complex one (Model B), on the same dataset. Performance metrics include Mean Squared Error (MSE), training loss, and validation loss. Results show Model A is more stable and has a lower validation loss, indicating better prediction accuracy.

## Conclusion
The proposed LSTM network is a promising, dependable, and computationally efficient technique for predicting nonlinear structural responses. The methodology demonstrates significant potential for assessing seismic risk and reliability analysis, addressing technical challenges such as accurate modeling, computational efficiency, data availability, and regional variability in ground motion characteristics.
