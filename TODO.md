# TODO

Trocamos uma das datasets? O prof queixou-se porque de facto as datasets são parecidas em estrutura.

* [ ] Trocar Dataset?
* [ ] Adicionar datasets a source/artifacts/data

-----

* [ ] Adicionar o PDF da proposta à pasta deliveries/

-----

ETL fazeado
Não é preciso gastar mais do que 6h no total nesta parte.  

Será mais importante validar os tipos das colunas (string/bool/etc), e tomar decisões de como vão ficar.  

É depois é analisar a falta de dados. O threshold está nos 10%. Se faltarem mais do que 10% de dados, remove-se a coluna. Não é preciso estar com muito mais. Se se vir que é fácil de fazer imputação a uma variável em especifico, seria interessante. Imputação com KNN é Boilerplate.  

Ver também as distribuições das variávels, mas não perder muito tempo nesta parte.  

As correlações entre variáveis possívelmente vão chatear bastante no SHAP e no LIME, visto que estão correlacionadas, vou fazer uma conjetura na medida em que o peso das variáveis na decisão feita pelo modelo está distrubuida por todas as variáveis, e se há duas que contribuem com info muito parecida, o peso delas vai estar mais ou menos dividido. Seria uma questão interessante a ver. ((Adicionado no Report.md))

Depois é ver o que fazemos com o que temos.

* [ ] Column types
* [ ] Missing Data
* [ ] Distribution
* [ ] Correlations
* [ ] Next Steps

----

Ler uns papers e apontar no Report.   
Eu queria ver se ia-se já adicionando nem que fossem só notas no Report.md, de forma a que depois para fazer o relatorio final, fosse mais fácil de (d)escrever o trabalho.

* [ ] Ler Papers
  * [ ] LIME-1 (Why should i trust you?: Explaining the predictions of any classifier.)
  * [ ] SHAP-1 (A Unified Approach to Interpretin ModelPredictions)