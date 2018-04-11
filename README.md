# Multiclass-Classification-Iris-Flower-Data-Set

After using logistic regression for binomial classification on news data <a href="http://rubyshrestha.com.np/2018/03/31/sentiment-analysis/" target="_blank">[blog: here]</a>, I wanted to explore the possibility of logistic regression in case of multiclass classification. Hence, I decided to use Iris Flower Data Set available in Kaggle which has three distinct classes for output variable. Kaggle is a great place for predictive modeling or data mining enthusiasts since one can get access to various kinds of data within its data realm; there are small to medium to even large sets of data, which one can use either for practice or to involve in the competitions out there or for the purpose of learning. I am using the flower data set for the purpose of gaining and sharing knowledge.

Although logistic regression is originally developed for binomial classification where the output variable is dichotomous in nature, it can be used in scenario where the output variable is trichotomous or polychotomous (with more than two discrete values) as well. There are schemes such as multinomial and one –versus-rest that can be used to generalize logistic regression to make it suitable for multi-class classification.

This git post is related to the one-versus-rest scheme of using logistic regression for multi-class classification. 

In one-versus-rest (ovr) scheme of logistic regression, from the number of possible values of dependent/ output variable, for each distinct group of two classes, a logistic regression is fitted. This means, if we have output variable that can have three possible values A, B and C, we will have three logistic regressions fitted, one for each group AB, BC and AC. When testing a test data x, its output from each fitted logistic regression is obtained and the class with highest score will be considered the destination class for the test data, since the score is actually the probability of falling into that class given the values of x.

Understanding ovr logistic regression based multi-class classification will be easier if we get our hands dirty with an example implementation. Hence, this git post.

In order to better understand the implementation and visualizations, follow the blog post: <a href='http://rubyshrestha.com.np/2018/04/08/logistic-regression-multiclass/' target='_blank'>here</a>.
